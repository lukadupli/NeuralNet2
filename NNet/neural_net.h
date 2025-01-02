#pragma once

#include "helpers.h"
#include "layer.h"
#include <iostream>
#include <fstream>

namespace NNet {
	class NeuralNet {
	private:
		d_F_vd_vd LossFunc;
		vd_F_vd_vd LossDeriv;

		int in_sz, out_sz;
		std::vector<Layer*> layers;
	public:
		NeuralNet(int input_sz, const std::vector<Layer*>& layers, d_F_vd_vd LossFunc, vd_F_vd_vd LossDeriv, d_F RandGen = DefaultRandom);
		NeuralNet(const NeuralNet& other);
		NeuralNet(std::istream& istr);
		NeuralNet(const std::string& path);
		~NeuralNet();

		d_F_vd_vd GetLossFunc() const;
		vd_F_vd_vd GetLossDeriv() const;

		int InSize() const;
		int OutSize() const;

		std::vector<Layer*> LayersCopy() const;

		Eigen::VectorXd Query(const Eigen::VectorXd& in);
		Eigen::VectorXd Query(const std::vector<double>& in);

		Eigen::VectorXd BackQuery(const Eigen::VectorXd& grads);

		double Fit(const Eigen::VectorXd& in, const Eigen::VectorXd& target);
		double Fit(const std::vector<double>& in, const std::vector<double>& target);

		std::istream& Load(std::istream& istr);
		void Load(const std::string& path);

		std::ostream& Save(std::ostream& ostr) const;
		void Save(const std::string& path) const;
	};
}