#include "pch.h"
#include "helpers.h"

namespace NNet {
    const double PI = acos(-1);

    double Scale(double val, double mini1, double maxi1, double mini2, double maxi2){
        if (mini1 == maxi1 || mini2 == maxi2) throw Exception("NNet::Scale: Minimum and maximum values must differ!");

        return (val - mini1) * (maxi2 - mini2) / (maxi1 - mini1) + mini2;
    }

    ///FFT of a vector padded with zeros so that its size is a power of 2
    ///d = 1 (default) performs FFT, d = -1 performs inverse FFT for vectors whose size is a power of 2
    Eigen::VectorXcd FFT(const Eigen::VectorXcd& v, int d) {
        int n = 1;
        while (n < v.size()) n *= 2;

        Eigen::VectorXcd r(n);

        for (int k = 0; k < n; k++) {
            int b = 0;
            for (int z = 1; z < n; z *= 2) {
                b *= 2;
                if (k & z) b++;
            }
            r[b] = (k < v.size()) ? v[k] : 0;
        }

        for (int m = 2; m <= n; m *= 2) {
            std::complex<double> wm = exp(std::complex<double>{0, d * 2 * PI / m});
            for (int k = 0; k < n; k += m) {
                std::complex<double> w = 1;
                for (int j = 0; j < m / 2; j++) {
                    std::complex<double> a = r[k + j], b = w * r[k + j + m / 2];
                    r[k + j] = a + b;
                    r[k + j + m / 2] = a - b;
                    w *= wm;
                }
            }
        }

        if (d == -1) r /= n;
        return r;
    }

    ///FFT of a Eigen::MatrixXd padded with zeros so that its dimensions are powers of 2
    ///d = 1 (default) performs FFT, d = -1 performs inverse FFT for a matrix whose dimensions are powers of 2
    Eigen::MatrixXcd FFT2(const Eigen::MatrixXcd& mat, int d) {
        int n = 1;
        while (n < mat.rows()) n *= 2;
        int m = 1;
        while (m < mat.cols()) m *= 2;

        Eigen::MatrixXcd r(n, m);
        r.fill(0);
        r.topLeftCorner(mat.rows(), mat.cols()) = mat;

        for (int i = 0; i < n; i++) r.row(i) = FFT(r.row(i), d);
        for (int i = 0; i < m; i++) r.col(i) = FFT(r.col(i), d);

        return r;
    }

    ///Computes convolution of two matrices using FFT2 function
    Eigen::MatrixXd Convolve2D(const Eigen::MatrixXd& signal, const Eigen::MatrixXd& mask) {
        int r = signal.rows() + mask.rows() - 1;
        int c = signal.cols() + mask.cols() - 1;

        Eigen::MatrixXd sig(r, c);
        sig.fill(0);
        sig.topLeftCorner(signal.rows(), signal.cols()) = signal;

        Eigen::MatrixXd mas(r, c);
        mas.fill(0);
        mas.topLeftCorner(mask.rows(), mask.cols()) = mask;

        Eigen::MatrixXcd res = FFT2(FFT2(sig).cwiseProduct(FFT2(mas)), -1);
        Eigen::MatrixXd ret(r, c);

        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) ret(i, j) = res(i, j).real();
        }

        return ret;
    }

    std::vector<Eigen::MatrixXd> VecTo3D(const Eigen::VectorXd& v, int d, int h, int w) {
        if (v.size() != d * h * w) throw Exception("VecTo3D: given vector doesn't match given dimensions!");
        std::vector<Eigen::MatrixXd> ret;
        for (int i = 0; i < d; i++) {
            ret.push_back(Eigen::MatrixXd(h, w));
            for (int j = 0; j < h; j++) {
                for (int k = 0; k < w; k++) ret[i](j, k) = v(i * h * w + j * w + k);
            }
        }

        return ret;
    }
    Eigen::VectorXd ThreeDToVec(const std::vector<Eigen::MatrixXd>& t) {
        int d = t.size(), h = t[0].rows(), w = t[0].cols();
        Eigen::VectorXd ret(d * h * w);
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < h; j++) {
                for (int k = 0; k < w; k++) ret(i * h * w + j * w + k) = t[i](j, k);
            }
        }

        return ret;
    }

    Eigen::VectorXd Sigmoid(const Eigen::VectorXd& in) {
        Eigen::VectorXd ret(in.size());

        for (int i = 0; i < in.size(); i++) ret(i) = 1 / (1 + exp(-in(i)));

        return ret;
    }
    Eigen::MatrixXd SigmoidDeriv(const Eigen::VectorXd& in) {
        Eigen::VectorXd sigmoid = Sigmoid(in);
        Eigen::MatrixXd ret(in.size(), in.size());

        for (int i = 0; i < ret.rows(); i++) {
            for (int j = 0; j < ret.cols(); j++) ret(i, j) = (i == j) ? sigmoid(i) * (1 - sigmoid(i)) : 0;
        }

        return ret;
    }

    Eigen::VectorXd Tanh(const Eigen::VectorXd& in) {
        Eigen::VectorXd ret = in;
        for (auto& e : ret) e = tanh(e);

        return ret;
    }
    Eigen::MatrixXd TanhDeriv(const Eigen::VectorXd& in) {
        Eigen::VectorXd tanh = Tanh(in);
        Eigen::MatrixXd ret(in.size(), in.size());

        for (int i = 0; i < ret.rows(); i++) {
            for (int j = 0; j < ret.cols(); j++) ret(i, j) = (i == j) ? 1 - tanh(i) * tanh(i) : 0;

        }

        return ret;
    }

    Eigen::VectorXd ReLU(const Eigen::VectorXd& in) {
        Eigen::VectorXd ret = in;

        for (auto& e : ret) e = std::max(0., e);

        return ret;
    }

    Eigen::MatrixXd ReLUDeriv(const Eigen::VectorXd& in) {
        Eigen::MatrixXd ret(in.size(), in.size());

        for (int i = 0; i < ret.rows(); i++) {
            for (int j = 0; j < ret.cols(); j++) ret(i, j) = (i == j) ? (in(i) > 0) : 0;
        }

        return ret;
    }

    Eigen::VectorXd Softmax(const Eigen::VectorXd& in) {
        double sum = 0.;
        for (auto x : in) sum += exp(x);

        Eigen::VectorXd ret = in;
        for (auto& x : ret) x = exp(x) / sum;

        return ret;
    }

    Eigen::MatrixXd SoftmaxDeriv(const Eigen::VectorXd& in) {
        Eigen::VectorXd softmax = Softmax(in);

        Eigen::MatrixXd ret(in.size(), in.size());

        for (int i = 0; i < in.size(); i++) {
            for (int j = 0; j < in.size(); j++) {
                if (j == i) ret(i, j) = softmax(i) * (1 - softmax(i));
                else ret(i, j) = -softmax(i) * softmax(j);
            }
        }
        return ret;
    }

    Eigen::VectorXd SqLossDeriv(const Eigen::VectorXd& out, const Eigen::VectorXd& target) {
        if (out.size() != target.size()) throw Exception("Sq_Loss_Deriv : out and target sizes don't match");

        Eigen::VectorXd ret(out.size());
        for (int i = 0; i < out.size(); i++) ret(i) = out(i) - target(i);

        return ret;
    }

    double SqLoss(const Eigen::VectorXd& out, const Eigen::VectorXd& target)
    {
        if (out.size() != target.size()) throw Exception("Sq_Loss : out and target sizes don't match");

        double ret = 0;
        for (int i = 0; i < out.size(); i++) ret += (out(i) - target(i)) * (out(i) - target(i));

        return ret;
    }

    double DefaultRandom() {
        static std::random_device rd;
        std::mt19937 gen(rd());

        std::uniform_real_distribution<double> normal { -1, 1 };
        return normal(gen);
    }

    double CrossEntropyLoss(const Eigen::VectorXd& out, const Eigen::VectorXd& target) {
        if (out.size() != target.size()) throw Exception("Cross_Entropy_Loss : out and target sizes don't match");

        double ret = 0.;

        for (int i = 0; i < out.size(); i++) {
            if (target(i) == 1) ret = -log(out(i));
        }

        return ret;
    }

    Eigen::VectorXd CrossEntropyLossDeriv(const Eigen::VectorXd& out, const Eigen::VectorXd& target) {
        if (out.size() != target.size()) throw Exception("Cross_Entropy_Loss_Deriv : out and target sizes don't match");

        Eigen::VectorXd ret(out.size());

        for (int i = 0; i < out.size(); i++) {
            ret(i) = target(i) ? -1. / out(i) : 0;
        }

        return ret;
    }

    double MaxPool(const Eigen::MatrixXd& mat) {
        return mat.maxCoeff();
    }
    Eigen::MatrixXd MaxPoolDeriv(const Eigen::MatrixXd& mat, double grad) {
        Eigen::MatrixXd ret = Eigen::MatrixXd::Zero(mat.rows(), mat.cols());

        double maxi = mat.maxCoeff();

        for (int x = 0; x < mat.rows(); x++) {
            for (int y = 0; y < mat.cols(); y++) {
                if (mat(x, y) == maxi) ret(x, y) = grad;
            }
        }

        return ret;
    }

    double AvgPool(const Eigen::MatrixXd& mat) {
        return mat.sum() / (mat.rows() * mat.cols());
    }
    Eigen::MatrixXd AvgPoolDeriv(const Eigen::MatrixXd& mat, double grad) {
        Eigen::MatrixXd ret{ mat.rows(), mat.cols() };

        ret.fill(grad);

        return ret;
    }

    std::vector<vd_F_vd> ActDecode{ nullptr, Sigmoid, Tanh, ReLU, Softmax };
    std::map<vd_F_vd, int> ActEncode{
        {nullptr, 0},
        {Sigmoid, 1},
        {Tanh, 2},
        {ReLU, 3},
        {Softmax, 4}
    };
    std::vector<md_F_vd> ActDerivDecode{ nullptr, SigmoidDeriv, TanhDeriv, ReLUDeriv, SoftmaxDeriv };
    std::map<md_F_vd, int> ActDerivEncode{
        {nullptr, 0},
        {SigmoidDeriv, 1},
        {TanhDeriv, 2},
        {ReLUDeriv, 3},
        {SoftmaxDeriv, 4}
    };

    std::vector<d_F_vd_vd> LossDecode{ nullptr, SqLoss, CrossEntropyLoss };
    std::map<d_F_vd_vd, int> LossEncode{
        {nullptr, 0},
        {SqLoss, 1},
        {CrossEntropyLoss, 2}
    };
    std::vector<vd_F_vd_vd> LossDerivDecode{ nullptr, SqLossDeriv, CrossEntropyLossDeriv };
    std::map<vd_F_vd_vd, int> LossDerivEncode{
        {nullptr, 0},
        {SqLossDeriv, 1},
        {CrossEntropyLossDeriv, 2}
    };

    std::vector<d_F_md> PoolDecode{ nullptr, MaxPool, AvgPool };
    std::map<d_F_md, int> PoolEncode{
        {nullptr, 0},
        {MaxPool, 1},
        {AvgPool, 2}
    };
    std::vector<md_F_md_d> PoolDerivDecode{ nullptr, MaxPoolDeriv, AvgPoolDeriv };
    std::map<md_F_md_d, int> PoolDerivEncode{
        {nullptr, 0},
        {MaxPoolDeriv, 1},
        {AvgPoolDeriv, 2}
    };

    std::istream& operator>>(std::istream& str, vd_F_vd& func)
    {
        int id;
        str >> id;

        func = ActDecode[id];

        return str;
    }
    std::ostream& operator<<(std::ostream& str, const vd_F_vd& func)
    {
        str << ActEncode[func] << ' ';
        return str;
    }

    std::istream& operator>>(std::istream& str, md_F_vd& func)
    {
        int id;
        str >> id;

        func = ActDerivDecode[id];

        return str;
    }
    std::ostream& operator<<(std::ostream& str, const md_F_vd& func)
    {
        str << ActDerivEncode[func] << ' ';
        return str;
    }

    std::istream& operator>>(std::istream& str, d_F_vd_vd& func)
    {
        int id;
        str >> id;

        func = LossDecode[id];

        return str;
    }
    std::ostream& operator<<(std::ostream& str, const d_F_vd_vd& func)
    {
        str << LossEncode[func] << ' ';
        return str;
    }

    std::istream& operator>>(std::istream& str, vd_F_vd_vd& func)
    {
        int id;
        str >> id;

        func = LossDerivDecode[id];

        return str;
    }
    std::ostream& operator<<(std::ostream& str, const vd_F_vd_vd& func)
    {
        str << LossDerivEncode[func] << ' ';
        return str;
    }

    std::istream& operator>>(std::istream& str, d_F_md& func) {
        int id;
        str >> id;

        func = PoolDecode[id];

        return str;
    }
    std::ostream& operator<<(std::ostream& str, const d_F_md& func) {
        str << PoolEncode[func] << ' ';
        return str;
    }

    std::istream& operator>>(std::istream& str, md_F_md_d& func) {
        int id;
        str >> id;

        func = PoolDerivDecode[id];

        return str;
    }
    std::ostream& operator<<(std::ostream& str, const md_F_md_d& func) {
        str << PoolDerivEncode[func] << ' ';
        return str;
    }
}
