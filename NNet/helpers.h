#pragma once

#include <vector>
#include <map>
#include <algorithm>
#include <random>

#include <Eigen/Dense>
#include "errors.h"

namespace NNet {
    double Scale(double val, double mini1, double maxi1, double mini2, double maxi2);

    ///FFT of a vector padded with zeros so that its size is a power of 2
    ///d = 1 (default) performs FFT, d = -1 performs inverse FFT for vectors whose size is a power of 2
    Eigen::VectorXcd FFT(const Eigen::VectorXcd& v, int d = 1);

    ///FFT of a matrix padded with zeros so that its dimensions are powers of 2
    ///d = 1 (default) performs FFT, d = -1 performs inverse FFT for a matrix whose dimensions are powers of 2
    Eigen::MatrixXcd FFT2(const Eigen::MatrixXcd& mat, int d = 1);

    ///Computes convolution of two matrices using FFT2 function
    Eigen::MatrixXd Convolve2D(const Eigen::MatrixXd& signal, const Eigen::MatrixXd& mask);

    std::vector<Eigen::MatrixXd> VecTo3D(const Eigen::VectorXd& v, int d, int h, int w);
    Eigen::VectorXd ThreeDToVec(const std::vector<Eigen::MatrixXd>& t);

    typedef double(*d_F)();
    double DefaultRandom();

    // --------------- Act --------------- //
    typedef Eigen::VectorXd(*vd_F_vd)(const Eigen::VectorXd&);

    Eigen::VectorXd Sigmoid(const Eigen::VectorXd& x);
    Eigen::VectorXd Tanh(const Eigen::VectorXd& x);
    Eigen::VectorXd ReLU(const Eigen::VectorXd& x);
    Eigen::VectorXd Softmax(const Eigen::VectorXd& in);

    std::istream& operator>>(std::istream& str, vd_F_vd& func);
    std::ostream& operator<<(std::ostream& str, const vd_F_vd& func);

    // --------------- ActDeriv --------------- //
    typedef Eigen::MatrixXd(*md_F_vd)(const Eigen::VectorXd&);

    Eigen::MatrixXd SigmoidDeriv(const Eigen::VectorXd& x);
    Eigen::MatrixXd TanhDeriv(const Eigen::VectorXd& x);
    Eigen::MatrixXd ReLUDeriv(const Eigen::VectorXd& x);
    Eigen::MatrixXd SoftmaxDeriv(const Eigen::VectorXd& in);

    std::istream& operator>>(std::istream& str, md_F_vd& func);
    std::ostream& operator<<(std::ostream& str, const md_F_vd& func);

    // --------------- Loss --------------- //
    typedef double(*d_F_vd_vd)(const Eigen::VectorXd&, const Eigen::VectorXd&);

    double SqLoss(const Eigen::VectorXd& out, const Eigen::VectorXd& target);
    double CrossEntropyLoss(const Eigen::VectorXd& out, const Eigen::VectorXd& target);

    std::istream& operator>>(std::istream& str, d_F_vd_vd& func);
    std::ostream& operator<<(std::ostream& str, const d_F_vd_vd& func);

    // --------------- LossDeriv --------------- //
    typedef Eigen::VectorXd(*vd_F_vd_vd)(const Eigen::VectorXd&, const Eigen::VectorXd&);

    Eigen::VectorXd SqLossDeriv(const Eigen::VectorXd& out, const Eigen::VectorXd& target);
    Eigen::VectorXd CrossEntropyLossDeriv(const Eigen::VectorXd& out, const Eigen::VectorXd& target);

    std::istream& operator>>(std::istream& str, vd_F_vd_vd& func);
    std::ostream& operator<<(std::ostream& str, const vd_F_vd_vd& func);

    // --------------- Pool --------------- //
    typedef double(*d_F_md)(const Eigen::MatrixXd&);

    double MaxPool(const Eigen::MatrixXd& mat);
    double AvgPool(const Eigen::MatrixXd& mat);

    std::istream& operator>>(std::istream& str, d_F_md& func);
    std::ostream& operator<<(std::ostream& str, const d_F_md& func);

    // --------------- PoolDeriv --------------- //

    typedef Eigen::MatrixXd(*md_F_md_d)(const Eigen::MatrixXd&, double);

    Eigen::MatrixXd MaxPoolDeriv(const Eigen::MatrixXd& mat, double grad);
    Eigen::MatrixXd AvgPoolDeriv(const Eigen::MatrixXd& mat, double grad);

    std::istream& operator>>(std::istream& str, md_F_md_d& func);
    std::ostream& operator<<(std::ostream& str, const md_F_md_d& func);

    // ----------------- END ----------------- //

    template <typename T>
    inline Eigen::VectorX<T> Vec2Eig(const std::vector<T>& v) {
        Eigen::VectorX<T> ret{ v.size() };
        for (int i = 0; i < v.size(); i++) ret(i) = v[i];
        return ret;
    }
}