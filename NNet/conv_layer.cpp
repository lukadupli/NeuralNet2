#include "pch.h"
#include "conv_layer.h"

namespace NNet {
	void ConvL::CalcOutSizes() {
		out_d = in_d * kernel_d;
		if (pad == SAME) { out_h = in_h; out_w = out_h; }
		else if (pad == VALID) { out_h = in_h - kernel_h + 1; out_w = in_w - kernel_w + 1; }
	}

	ConvL::ConvL(double lrate_, int input_h, int input_w, int kernel_d, int kernel_h, int kernel_w, Padding pad) : 
		in_h(input_h), in_w(input_w), kernel_d(kernel_d), kernel_h(kernel_h), kernel_w(kernel_w), pad(pad)
	{
		lrate = lrate_;
		id = "Conv";
	}
	ConvL::ConvL(const ConvL& other) {
		in_d = other.InDepth();
		in_h = other.InHeight();
		in_w = other.InWidth();

		kernels = other.Kernels();
		kernel_d = kernels.size();
		kernel_h = kernels[0].rows();
		kernel_w = kernels[0].cols();

		lrate = other.LRate();
		pad = other.GetPadding();

		CalcOutSizes();

		id = "Conv";
	}
	ConvL::ConvL(std::istream& istr) {
		id = "Conv";
		Read(istr);
	}
	
	int ConvL::InDepth() const { return in_d; }
	int ConvL::InHeight() const { return in_h; }
	int ConvL::InWidth() const { return in_w; }

	Padding ConvL::GetPadding() const { return pad; }
	std::vector<Eigen::MatrixXd> ConvL::Kernels() const { return kernels; }

	void ConvL::SetInputSize(int in_sz) {
		if (in_sz % (in_h * in_w)) throw Exception("ConvL::SetInputSize: Make sure total input size is divisible by the product of input height and width!");
		in_d = in_sz / (in_h * in_w);
		CalcOutSizes();
	}
	void ConvL::InitParams(d_F GenFunc) {
		kernels.clear();
		for (int i = 0; i < kernel_d; i++) {
			kernels.push_back(Eigen::MatrixXd(kernel_h, kernel_w));
			for (int j = 0; j < kernel_h; j++) {
				for (int k = 0; k < kernel_w; k++) kernels[i](j, k) = GenFunc();
			}
		}
	}
	int ConvL::OutSize() const {
		return out_d * out_h * out_w;
	}

	Eigen::VectorXd ConvL::Forward(const Eigen::VectorXd& in) {
		if (in.size() != in_d * in_h * in_w) throw Exception("ConvL::Forward: Input size doesn't match!");
		auto t = VecTo3D(in, in_d, in_h, in_w);
		cache = t;

		std::vector<Eigen::MatrixXd> ret;
		for (int i = 0; i < in_d; i++) {
			for (int j = 0; j < kernel_d; j++) {
				Eigen::MatrixXd res = Convolve2D(t[i], kernels[j]);
				if (pad == SAME) ret.push_back(res.block((kernel_h - 1) / 2, (kernel_w - 1) / 2, out_h, out_w));
				else if (pad == VALID) ret.push_back(res.block(kernel_h - 1, kernel_w - 1, out_h, out_w));
			}
		}

		return ThreeDToVec(ret);
	}
	Eigen::VectorXd ConvL::Backward(const Eigen::VectorXd& grads) {
		if (grads.size() != out_d * out_h * out_w) throw Exception("ConvL::Backward: Gradient list is not the right size!");
		auto g = VecTo3D(grads, out_d, out_h, out_w);

		std::vector<Eigen::MatrixXd> ret;
		for (int i = 0; i < in_d; i++) {
			ret.push_back(Eigen::MatrixXd::Zero(in_h, in_w));
			for (int j = 0; j < kernel_d; j++) {
				auto res = Convolve2D(g[i * kernel_d + j], kernels[j].reverse());
				if (pad == SAME) ret[i] += res.block((kernel_h - 1) / 2, (kernel_w - 1) / 2, in_h, in_w);
				else if (pad == VALID) ret[i] += res;
			}
		}

		for (int j = 0; j < kernel_d; j++) {
			for (int i = 0; i < in_d; i++) {
				auto res = Convolve2D(g[i * kernel_d + j], cache[i]);
				int midx = res.rows() / 2 - kernel_h / 2;
				int midy = res.cols() / 2 - kernel_w / 2;

				kernels[j] -= lrate * res.block(midx, midy, kernel_h, kernel_w);
			}
		}

		return ThreeDToVec(ret);
	}

	std::istream& ConvL::Read(std::istream& istr) {
		istr >> in_d >> in_h >> in_w >> kernel_d >> kernel_w >> kernel_h;
		int a;
		istr >> a;
		if (a < 0 || a >= 2) throw Exception("ConvL::Read: Invalid data given to read from stream!");
		pad = static_cast<Padding>(a);

		CalcOutSizes();

		kernels.clear();
		for (int i = 0; i < kernel_d; i++) {
			kernels.push_back(Eigen::MatrixXd{ kernel_h, kernel_w });
			for (int j = 0; j < kernel_h; j++) {
				for (int k = 0; k < kernel_w; k++) istr >> kernels[i](j, k);
			}
		}

		return istr;
	}

	std::ostream& ConvL::Write(std::ostream& ostr) const {
		ostr << id << '\n';
		ostr << in_d << ' ' << in_h << ' ' << in_w << '\n' << kernel_d << ' ' << kernel_w << ' ' << kernel_h << '\n' << pad << '\n';

		for (int i = 0; i < kernel_d; i++) ostr << kernels[i] << '\n';

		return ostr;
	}
}