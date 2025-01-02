#define EXCLUDE
#ifndef EXCLUDE

#include <iostream>
#include <filesystem>

#include "../NNet/neural_net.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace NNet;
using namespace Eigen;

const string DATASET = R"(C:\Users\lukad\Desktop\math_symbols\)";
const string NN_DIR = R"(C:\Users\lukad\Desktop\math_symbols\nn_models\)";
const int TRESHOLD = 230;

std::vector<std::string> GetFilenames(const string& folder)
{
    namespace stdfs = std::filesystem;

    if (!stdfs::exists(folder)) {
        std::cerr << "GetFilenames( " << folder << "): no such path exists!\n";
        throw std::runtime_error("");
    }

    std::vector<std::string> filenames;
    const stdfs::directory_iterator end{};

    for (stdfs::directory_iterator iter{ folder }; iter != end; ++iter)
    {
        if (stdfs::is_regular_file(*iter))
            filenames.push_back(iter->path().string());
    }

    return filenames;
}
void Treshold(cv::Mat& img, const uchar treshold) {
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            auto& val = img.at<uchar>(i, j);
            val = (val < treshold) ? 0 : 255;
        }
    }
}
Eigen::MatrixXd Treshold(const Eigen::MatrixXd& mat, const double treshold) {
    Eigen::MatrixXd ret{ mat.rows(), mat.cols() };

    for (int i = 0; i < mat.rows(); i++) {
        for (int j = 0; j < mat.cols(); j++) {
            ret(i, j) = (mat(i, j) < treshold) ? 0 : 1;
        }
    }

    return ret;
}
Eigen::MatrixXd cvMat2MatrixXd(const cv::Mat& img) {
    Eigen::MatrixXd ret{ img.rows, img.cols };

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) ret(i, j) = img.at<uchar>(i, j);
    }

    return ret;
}

vector<string> decode{ "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "mul", "div", "=", "(", ")" };
vector<vector<Eigen::MatrixXd>> LoadData() {
    vector<vector<Eigen::MatrixXd>> ret;

    for (int id = 0; id < decode.size(); id++) {
        auto fnames = GetFilenames(DATASET + decode[id]);

        ret.push_back(vector<Eigen::MatrixXd>{});
        for (int i = 0; i < fnames.size(); i++) {
            auto fname = fnames[i];
            cv::Mat img = cv::imread(fname, cv::IMREAD_GRAYSCALE);

            ret.back().push_back(cvMat2MatrixXd(img));
            for (int i = 0; i < ret.back().back().rows(); i++) {
                for (int j = 0; j < ret.back().back().cols(); j++) {
                    ret.back().back()(i, j) = Scale(ret.back().back()(i, j), 0, 255, 0, 1);
                }
            }

            //ret.back().back() = Treshold(ret.back().back(), 0.7);
        }
    }

    return ret;
}

void Train(NeuralNet& net, const string& modelpath, const vector<vector<Eigen::MatrixXd>>& data, int prefix = -1, const std::vector<int> classes = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }) {
    vector<int> ptr(data.size());
    vector<int> order;

    for (int i : classes) {
        int lim = (prefix > 0) ? prefix : data[i].size();
        for (int j = 0; j < std::min(lim, (int)data[i].size()); j++) {
            order.push_back(i);
        }
    }

    int epochs;
    std::cout << "No. of epochs: "; cin >> epochs;

    int step, save_step;
    std::cout << "How often would you like to be informed of progress? (image number): "; cin >> step;
    std::cout << "How often would you like to save your NN? (image number): "; cin >> save_step;

    for (int ep = 1; ep <= epochs; ep++) {
        std::cout << "---------- Epoch " << ep << " ----------\n";
        for (auto& e : ptr) e = 0;

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        shuffle(order.begin(), order.end(), std::default_random_engine(seed));

        double loss = 0;
        for (int i = 0; i < order.size(); i++) {
            Eigen::VectorXd tar = Eigen::VectorXd::Zero(data.size());
            tar[order[i]] = 1;

            Eigen::MatrixXd in = data[order[i]][ptr[order[i]]++];

            //auto result = net.Query(ThreeDToVec(vector<Eigen::MatrixXd>{in}));
            loss += net.Fit(ThreeDToVec(vector<Eigen::MatrixXd>{in}), tar);

            if ((i + 1) % step == 0) {
                std::cout << "Passed " << i + 1 << " images, average loss is: " << loss / step << '\n';

                /*if (loss / step < 0.25) {
                    net.Universal_Lrate(0.001);
                    net.Universal_Bias_Lrate(0.002);
                }*/

                /*std::cout << Treshold(in, 0.7) << '\n' << result << "\n\n";

                int maxind = -1;
                double maxi = -1e9;

                for (int iter = 0; iter < result.size(); iter++) {
                    if (result[iter] > maxi) {
                        maxind = iter;
                        maxi = result[iter];
                    }
                }

                std::std::cout << "Last target: " << order[i] << ", net prediction: " << maxind << '\n';*/

                if (isnan(loss)) {
                    std::cout << "Loss is NaN, breaking the training process...\n";
                    return;
                }

                loss = 0;
            }

            if ((i + 1) % save_step == 0) {
                std::cout << "Passed " << i + 1 << " images, saving your model to file: " << modelpath << " ... ";
                net.Save(modelpath);

                std::cout << "Done saving\n";
            }
        }

        std::cout << "Finished epoch " << ep << ", saving your model to file: " << modelpath << " ... ";
        net.Save(modelpath);
        std::cout << "Done saving\n";
    }
}
void Test(NeuralNet& net, const vector<vector<Eigen::MatrixXd>>& data, int train_pref = 0) {
    std::cout << "Testing...\n";

    int total_corr = 0, total_cnt = 0;
    for (int id = 0; id < data.size(); id++) {
        std::cout << "Symbol " << decode[id] << ":\n";

        int digit_corr = 0;
        for (int nes = train_pref; nes < data[id].size(); nes++) {
            auto input = data[id][nes];

            auto res = net.Query(ThreeDToVec(vector<Eigen::MatrixXd>{input}));

            double maxi = -1e9;
            int pred = -1;

            for (int i = 0; i < res.size(); i++) {
                if (res(i) > maxi) {
                    maxi = res(i);
                    pred = i;
                }
            }

            if (pred == id) digit_corr++;
        }

        total_corr += digit_corr;
        total_cnt += data[id].size() - train_pref;

        std::cout << "   Local accuracy: " << digit_corr << "/" << (data[id].size() - train_pref) << ", " << (double)digit_corr / (data[id].size() - train_pref) * 100 << "%\n";
        std::cout << "   Total accuracy until now: " << total_corr << "/" << total_cnt << ", " << (double)total_corr / total_cnt * 100 << "%\n\n";
    }
}

void Questionaire(NeuralNet& net) {
    while (true) {
        std::string fname;
        std::cin >> fname;

        if (fname == "break") break;

        cv::Mat img = cv::imread(fname, cv::IMREAD_GRAYSCALE);

        Eigen::MatrixXd input = cvMat2MatrixXd(img);

        std::cout << net.Query(ThreeDToVec(std::vector<Eigen::MatrixXd>{input})) << "\n\n";
    }
}

NeuralNet net
{ 
    59 * 59,
    {
    new ConvL(0.01, 59, 59, 4, 5, 5, SAME),
    new PoolL(59, 59, 4, 4, MaxPool, MaxPoolDeriv),
    new ActL(0.02, ReLU, ReLUDeriv),
    new ConvL(0.01, 15, 15, 4, 3, 3, SAME),
    new PoolL(15, 15, 2, 2, MaxPool, MaxPoolDeriv),
    new ActL(0.02, Tanh, TanhDeriv),
    new DenseL(0.01, 100),
    new ActL(0.02, Tanh, TanhDeriv),
    new DenseL(0.01, 17),
    new ActL(0.02, Softmax, SoftmaxDeriv),
    },
    CrossEntropyLoss,
    CrossEntropyLossDeriv
};

int main()
{
    std::cout << "Loading data...\n";
    auto data = LoadData();
    //net.Load(NN_DIR + "model7998.txt");

    Train(net, NN_DIR + "nnet_test2.txt", data, 1000);
    Test(net, data, 1000);

    Questionaire(net);
}

#endif