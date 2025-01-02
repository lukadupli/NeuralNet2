#pragma once

#include "helpers.h"
#include <Eigen/Dense>

namespace NNet {
	class Layer {
	protected:
		double lrate;

		int in_sz, out_sz;

		std::string id;
	public:
		virtual ~Layer() = default;

		virtual Layer* Clone() const = 0;

        std::string ID() const;
		double LRate() const;

		virtual void InitParams(d_F GenFunc) = 0;
		virtual void SetInputSize(int input_sz) = 0;

		virtual int OutSize() const = 0;

		virtual Eigen::VectorXd Forward(const Eigen::VectorXd&) = 0;
		virtual Eigen::VectorXd Backward(const Eigen::VectorXd&) = 0;

		virtual std::istream& Read(std::istream&) = 0;
		virtual std::ostream& Write(std::ostream&) const = 0;
	};

	template <typename LType> class LayerCRTP : public Layer{
		Layer* Clone() const override {
			return new LType(static_cast<const LType&>(*this));
		}
	};
}

std::istream& operator>>(std::istream&, NNet::Layer*&);
std::ostream& operator<<(std::ostream&, const NNet::Layer*&);

#include "dense_layer.h"
#include "act_layer.h"
#include "conv_layer.h"
#include "pool_layer.h"