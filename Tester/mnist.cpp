#include <iostream>
#include "../NNet/neural_net.h";
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
using namespace NNet;

typedef unsigned char ubyte;

int Read4ByteInt(std::istream& stream) {
    unsigned char buf[4];
    stream.read((char*)buf, 4);

    int ret = 0;
    for (int i = 0; i < 4; i++) {
        ret <<= 8;
        ret |= buf[i];
    }

    return ret;
}

std::vector<ubyte> ReadUbyteIdx1File(std::istream& stream) {
    for (int i = 0; i < 4; i++) stream.get();

    int size = Read4ByteInt(stream);

    std::vector<ubyte> ret;
    ret.resize(size);
    stream.read((char*)(&ret[0]), size);

    return ret;
}

std::vector<Eigen::MatrixXd>
ReadUbyteIdx3File(std::istream& stream) {
    for (int i = 0; i < 4; i++) stream.get();

    unsigned int size = Read4ByteInt(stream);
    unsigned int rows = Read4ByteInt(stream);
    unsigned int cols = Read4ByteInt(stream);

    std::vector<ubyte> bytes{ std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>() };

    std::vector<Eigen::MatrixXd> ret;
    for (int n = 0; n < size; n++) {
        ret.push_back(Eigen::MatrixXd(rows, cols));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) ret.back()(i, j) = NNet::Scale((double)bytes[n * rows * cols + i * cols + j], 0, 255, 0, 1);
        }
    }

    return ret;
}

std::vector<std::vector<ubyte>>
ReadAndFlattenUbyteIdx3File(std::istream& stream) {
    for (int i = 0; i < 4; i++) stream.get();

    unsigned int size = Read4ByteInt(stream);
    unsigned int rows = Read4ByteInt(stream);
    unsigned int cols = Read4ByteInt(stream);

    std::vector<ubyte> bytes{ std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>() };

    std::vector<std::vector<ubyte>> ret(size);
    for (int n = 0; n < size; n++) {
        ret[n].resize(rows * cols);
        for (int i = 0; i < rows * cols; i++) {
            ret[n][i] = bytes[n * rows * cols + i];
        }
    }

    return ret;
}

const string LOCATION = R"(C:\Users\lukad\Desktop\MNIST\)";
string FullPath(const string& str) {
    return LOCATION + str;
}

/*NeuralNet net
{
    28 * 28,
    {
    new ConvL(0.01, 28, 28, 4, 5, 5, SAME),
    new PoolL(28, 28, 4, 4, MaxPool, MaxPoolDeriv),
    new ActL(0.02, ReLU, ReLUDeriv),
    new DenseL(0.01, 100),
    new ActL(0.02, Tanh, TanhDeriv),
    new DenseL(0.01, 10),
    new ActL(0.02, Softmax, SoftmaxDeriv),
    },
    CrossEntropyLoss,
    CrossEntropyLossDeriv
};*/

NeuralNet net{
    28 * 28,
    {
        new ConvL(0.0001, 28, 28, 8, 3, 3, SAME),
        new PoolL(28, 28, 2, 2, MaxPool, MaxPoolDeriv),
        new ActL(0.02, ReLU, ReLUDeriv),
        new ConvL(0.0001, 14, 14, 8, 3, 3, SAME),
        new PoolL(14, 14, 2, 2, MaxPool, MaxPoolDeriv),
        new ActL(0.02, ReLU, ReLUDeriv),
        new DenseL(0.01, 100),
        new ActL(0.02, Sigmoid, SigmoidDeriv),
        new DenseL(0.01, 10),
        new ActL(0.02, Softmax, SoftmaxDeriv),
    },
    CrossEntropyLoss,
    CrossEntropyLossDeriv,
};

void Train(NeuralNet& net, const string& saveFile, const string& lossDump, const string& loadFile = "") {
    cout << "Loading data...\n";
    if (!loadFile.empty()) net.Load(FullPath(loadFile));

    ifstream labfile(FullPath("MNIST_dataset\\train-labels.idx1-ubyte"), std::ios::binary);
    auto labels = ReadUbyteIdx1File(labfile);
    labfile.close();

    ifstream imfile(FullPath("MNIST_dataset\\train-images.idx3-ubyte"), std::ios::binary);
    auto images = ReadUbyteIdx3File(imfile);
    imfile.close();

    int offset, how_many, step;
    cout << "Index of first image used for training (starting with 0): "; cin >> offset;
    cout << "How many images to take: "; cin >> how_many;
    cout << "How often would you like to be informed of progress (image number): "; cin >> step;
    cout << "Training...\n";

    if (offset + how_many > images.size()) {
        cout << "Not enough images in the dataset. Taking the whole dataset for training...\n";
        how_many = images.size() - offset;
    }

    vector<double> losses;
    double loss = 0;
    for (int i = offset; i < offset + how_many; i++) {
        VectorXd tar = VectorXd::Zero(10);
        tar[labels[i]] = 1;

        losses.push_back(net.Fit(ThreeDToVec(vector<MatrixXd>{images[i]}), tar));
        loss += losses.back();

        if ((i + 1) % step == 0) {
            cout << "Passed " << i + 1 << " images, average loss is: " << loss / step << '\n';
            if (isnan(loss)) {
                cout << "Loss is NaN, breaking the training process...\n";
                return;
            }

            loss = 0;
            net.Save(FullPath(saveFile));
        }
    }

    net.Save(FullPath(saveFile));

    ofstream csvfile(FullPath(lossDump));
    for (int i = 1; i <= how_many; i++) {
        csvfile << i << ", " << losses[i - 1] << '\n';
    }

    csvfile.close();
}

void Test(NeuralNet& net) {
    cout << "Loading data...\n";

    ifstream labfile(FullPath("MNIST_dataset\\test-labels.idx1-ubyte"), std::ios::binary);
    auto labels = ReadUbyteIdx1File(labfile);
    labfile.close();

    ifstream imfile(FullPath("MNIST_dataset\\test-images.idx3-ubyte"), std::ios::binary);
    auto images = ReadUbyteIdx3File(imfile);
    imfile.close();

    int correct = 0;
    cout << "Testing...\n";
    for (int i = 0; i < images.size(); i++) {
        auto out = net.Query(ThreeDToVec(vector<MatrixXd>{images[i]}));
        bool ok = 1;

        double maxi = -1; int ind = -1;
        for (int j = 0; j < 10; j++) {
            if (out(j) > maxi) {
                ind = j;
                maxi = out(j);
            }
        }

        if (ind == labels[i]) correct++;

        if ((i + 1) % 1000 == 0) {
            cout << "Passed " << i + 1 << " images, accuracy is: " << (double)correct / (i + 1) << '\n';
        }
    }
}

int main() {
    //Train(net, "MNIST_MITdemo.txt", "MNIST_MITdemo.csv");
    NeuralNet model{ FullPath("MNIST_MITdemo.txt") };
    Test(model);

    return 0;
}