#pragma once

#include "helpers.h"
#include "layer.h"

namespace NNet {
    class PoolL : public LayerCRTP<PoolL> {
    private:
        int dep, in_h, in_w, scan_h, scan_w, out_h, out_w;
        d_F_md PoolFunc;
        md_F_md_d PoolDeriv;

        std::vector<Eigen::MatrixXd> cache;

        void CalcOutSizes();
    public:
        PoolL(int in_h, int in_w, int scan_h, int scan_w, d_F_md PoolFunc, md_F_md_d PoolDeriv);
        PoolL(const PoolL& other);
        PoolL(std::istream& istr);

        int InHeight() const;
        int InWidth() const;
        int InDepth() const;

        int ScanHeight() const;
        int ScanWidth() const;

        d_F_md GetPoolFunc() const;
        md_F_md_d GetPoolDeriv() const;

        void InitParams(d_F GenFunc) override;
        void SetInputSize(int input_sz) override;

        int OutSize() const override;

        Eigen::VectorXd Forward(const Eigen::VectorXd& in) override;
        Eigen::VectorXd Backward(const Eigen::VectorXd& grads) override;

        std::istream& Read(std::istream& istr) override;
        std::ostream& Write(std::ostream& ostr) const override;
    };
}