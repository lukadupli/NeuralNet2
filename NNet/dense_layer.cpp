#include "pch.h"
#include "dense_layer.h"

namespace NNet {
	DenseL::DenseL(double lrate_, int out_sz_) { id = "Dense"; lrate = lrate_; out_sz = out_sz_; }
	DenseL::DenseL(const DenseL& other) {
		weights = other.Weights();

		lrate = other.LRate();
		in_sz = other.InSize();
		out_sz = other.OutSize();

		id = "Dense";
	}
	DenseL::DenseL(std::istream& istr) {
		id = "Dense";
		Read(istr);
	}

	void DenseL::SetInputSize(int input_sz) {
		in_sz = input_sz;
	}

	int DenseL::InSize() const { return in_sz; }
	int DenseL::OutSize() const { return out_sz; }

	void DenseL::InitParams(d_F GenFunc) {
		weights = Eigen::MatrixXd{ out_sz, in_sz };
		for (int i = 0; i < out_sz; i++) {
			for (int j = 0; j < in_sz; j++) weights(i, j) = GenFunc();
		}
	}

	Eigen::MatrixXd DenseL::Weights() const { return weights; }

	Eigen::VectorXd DenseL::Forward(const Eigen::VectorXd& in) {
		if (in.size() != in_sz) throw Exception("DenseL::Forward: Input sizes don't match");
		cache = in;
		return weights * in;
	}
	Eigen::VectorXd DenseL::Backward(const Eigen::VectorXd& grads) {
		if (grads.size() != out_sz) throw Exception("DenseL::Backward: Gradient vector size doesn't match");
		Eigen::VectorXd ret = weights.transpose() * grads;

		for (int i = 0; i < in_sz; i++) weights.col(i) -= lrate * cache(i) * grads;

		return ret;
	}

	std::istream& DenseL::Read(std::istream& istr) {
		istr >> in_sz >> out_sz >> lrate;

		weights = Eigen::MatrixXd{out_sz, in_sz};
		for (int i = 0; i < out_sz; i++) {
			for (int j = 0; j < in_sz; j++) istr >> weights(i, j);
		}

		return istr;
	}

	std::ostream& DenseL::Write(std::ostream& ostr) const {
		ostr << id << '\n' << in_sz << ' ' << out_sz << ' ' << lrate << '\n' << weights << '\n';
		return ostr;
	}
}