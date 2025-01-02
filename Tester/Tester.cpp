#define EXCLUDE
#ifndef EXCLUDE

#include "../NNet/neural_net.h";
#include <Eigen/Dense>

using namespace std;
using namespace NNet;

const string NN_DIR = R"(C:\Users\lukad\Desktop\math_symbols\nn_models\)";

NeuralNet net{
    5 * 5,
    {
       new ConvL(0.02, 5, 5, 2, 3, 3, SAME),
       new PoolL(5, 5, 2, 2, MaxPool, MaxPoolDeriv),
       new ActL(0.01, Tanh, TanhDeriv),
       new DenseL(0.02, 2),
       new ActL(0.01, Softmax, SoftmaxDeriv),
    },
    CrossEntropyLoss,
    CrossEntropyLossDeriv
};

vector<Eigen::MatrixXd> in{ Eigen::MatrixXd{5, 5} };
vector<double> out = { 0, 1 };

int main() {
    in.front() << 
        0.1, 0.2, 0.1, 0.2, 0.1,
        0.2, 0.1, 0.2, 0.1, 0.2,
        0.1, 0.2, 0.1, 0.2, 0.1,
        0.2, 0.1, 0.2, 0.1, 0.2,
        0.1, 0.2, 0.1, 0.2, 0.1;

    cout << in.front() << '\n';

    net.Load(NN_DIR + "dbg.txt");
    /*for (int i = 0; i < 100; i++) {
        cout << net.Fit(ThreeDToVec(in), Vec2Eig(out)) << '\n';
    }*/

    cout << "Result:\n" << net.Query(ThreeDToVec(in)) << '\n';
    //net.Save(NN_DIR + "dbg.txt");

    return 0;
}

#endif