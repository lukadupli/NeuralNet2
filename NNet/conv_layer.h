#pragma once

#include "layer.h"
#include "helpers.h"

namespace NNet {
	enum Padding{VALID, SAME};

	class ConvL : public LayerCRTP<ConvL> {
	private:
		int in_d, in_h, in_w;
		int out_d, out_h, out_w;
		int kernel_d, kernel_h, kernel_w;
		Padding pad;
		std::vector<Eigen::MatrixXd> kernels, cache;

		void CalcOutSizes();
	public:
		ConvL(double lrate, int input_h, int input_w, int kernel_d, int kernel_h, int kernel_w, Padding pad);
		ConvL(const ConvL& other);
		ConvL(std::istream& istr);

		int InDepth() const;
		int InHeight() const;
		int InWidth() const;

		Padding GetPadding() const;
		std::vector<Eigen::MatrixXd> Kernels() const;

		void SetInputSize(int in_sz) override;
		void InitParams(d_F GenFunc) override;
		int OutSize() const override;

		Eigen::VectorXd Forward(const Eigen::VectorXd& in) override;
		Eigen::VectorXd Backward(const Eigen::VectorXd& grads) override;

		std::istream& Read(std::istream& istr) override;
		std::ostream& Write(std::ostream& ostr) const override;
	};
}