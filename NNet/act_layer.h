#pragma once

#include "helpers.h"
#include "layer.h"

namespace NNet {
	class ActL : public LayerCRTP<ActL> {
	private:
		vd_F_vd ActFunc;
		md_F_vd ActDeriv;
		Eigen::VectorXd bias, cache;
	public:
		ActL(double lrate_, vd_F_vd ActFunc_, md_F_vd ActDeriv_);
		ActL(const ActL& other);
		ActL(std::istream& istr);
		~ActL() = default;

		vd_F_vd GetActFunc() const;
		md_F_vd GetActDeriv() const;

		Eigen::VectorXd Bias() const;

		void InitParams(d_F GenFunc) override;
		void SetInputSize(int input_sz) override;

		int InSize() const;
		int OutSize() const override;

		virtual Eigen::VectorXd Forward(const Eigen::VectorXd&) override;
		virtual Eigen::VectorXd Backward(const Eigen::VectorXd&) override;

		virtual std::istream& Read(std::istream&) override;
		virtual std::ostream& Write(std::ostream&) const override;
	};
}