#include "pch.h"
#include "pool_layer.h"

namespace NNet {
    void PoolL::CalcOutSizes() {
        out_h = (in_h - 1) / scan_h + 1;
        out_w = (in_w - 1) / scan_w + 1;
    }

    PoolL::PoolL(int in_h, int in_w, int scan_h, int scan_w, d_F_md PoolFunc, md_F_md_d PoolDeriv) :
        in_h(in_h), in_w(in_w), scan_h(scan_h), scan_w(scan_w), PoolFunc(PoolFunc), PoolDeriv(PoolDeriv) {
        id = "Pool";
        CalcOutSizes();
    }
    PoolL::PoolL(const PoolL& other) {
        in_h = other.InHeight();
        in_w = other.InWidth();
        dep = other.InDepth();

        scan_h = other.ScanHeight();
        scan_w = other.ScanWidth();

        PoolFunc = other.GetPoolFunc();
        PoolDeriv = other.GetPoolDeriv();

        id = "Pool";
        CalcOutSizes();
    }
    PoolL::PoolL(std::istream& istr) {
        Read(istr);
        id = "Pool";
    }

    int PoolL::InHeight() const { return in_h; }
    int PoolL::InWidth() const { return in_w; } 
    int PoolL::InDepth() const { return dep; }

    int PoolL::ScanHeight() const { return scan_h; }
    int PoolL::ScanWidth() const { return scan_w; }

    d_F_md PoolL::GetPoolFunc() const { return PoolFunc; }
    md_F_md_d PoolL::GetPoolDeriv() const { return PoolDeriv; }

    void PoolL::InitParams(d_F GenFunc) {}
    void PoolL::SetInputSize(int input_sz) { 
        if (input_sz % (in_h * in_w)) throw Exception("PoolL::SetInputSize: Make sure input size is divisible by in_h * in_w!");
        dep = input_sz / (in_h * in_w);
    }

    int PoolL::OutSize() const { return dep * out_w * out_h; }

    Eigen::VectorXd PoolL::Forward(const Eigen::VectorXd& in) {
        auto real = VecTo3D(in, dep, in_h, in_w);
        cache = real;

        std::vector<Eigen::MatrixXd> out;

        for (const Eigen::MatrixXd& mat : real) {
            out.push_back(Eigen::MatrixXd{out_h, out_w});
            for (int i = 0; i < in_h; i += scan_h) {
                for (int j = 0; j < in_w; j += scan_w)
                    out.back()(i / scan_h, j / scan_w) = PoolFunc(mat.block(i, j, std::min(scan_h, (int)mat.rows() - i), std::min(scan_w, (int)mat.cols() - j)));
            }
        }

        return ThreeDToVec(out);
    }

    Eigen::VectorXd PoolL::Backward(const Eigen::VectorXd& grads) {
        if (cache.empty()) throw Exception("PoolL::Backward: Backward without previous Forward!");
        auto real = VecTo3D(grads, dep, out_h, out_w);

        std::vector<Eigen::MatrixXd> ret;
        for (int z = 0; z < dep; z++) {
            ret.push_back(Eigen::MatrixXd{ in_h, in_w });
            for (int i = 0; i < in_h; i += scan_h) {
                for (int j = 0; j < in_w; j += scan_w) {
                    int bx = std::min(scan_h, (int)in_h - i), by = std::min(scan_w, (int)in_w - j);
                    ret.back().block(i, j, bx, by) = PoolDeriv(cache[z].block(i, j, bx, by), real[z](i / scan_h, j / scan_w));
                }
            }
        }

        return ThreeDToVec(ret);
    }

    std::istream& PoolL::Read(std::istream& istr) {
        cache.clear();
        istr >> dep >> in_h >> in_w >> scan_h >> scan_w >> PoolFunc >> PoolDeriv;
        CalcOutSizes();

        return istr;
    }

    std::ostream& PoolL::Write(std::ostream& ostr) const {
        ostr << id << '\n' << dep << ' ' << in_h << ' ' << in_w << '\n' << scan_h << ' ' << scan_w << '\n' << PoolFunc << PoolDeriv << '\n';

        return ostr;
    }
}