#pragma once

#include "helpers.h"
#include "layer.h"

namespace NNet {
	class DenseL : public LayerCRTP<DenseL> {
	private:
		Eigen::VectorXd cache;
		Eigen::MatrixXd weights;
	public:
		DenseL(double lrate_, int out_sz);
		DenseL(const DenseL& other);
		DenseL(std::istream& istr);
		~DenseL() = default;

		void InitParams(d_F GenFunc) override;
		void SetInputSize(int input_sz) override;

		int InSize() const;
		int OutSize() const override;

		Eigen::MatrixXd Weights() const;

		Eigen::VectorXd Forward(const Eigen::VectorXd& in) override;
		Eigen::VectorXd Backward(const Eigen::VectorXd& grads) override;

		std::istream& Read(std::istream& istr);
		std::ostream& Write(std::ostream& ostr) const;
	};
}
