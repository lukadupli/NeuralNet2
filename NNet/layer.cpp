#include "pch.h"
#include "layer.h"

namespace NNet {
	int Layer::OutSize() const { return out_sz; }
    std::string Layer::ID() const { return id; }
	double Layer::LRate() const { return lrate; }
}

std::istream& operator>>(std::istream& istr, NNet::Layer*& layer) {
	return layer->Read(istr);
}
std::ostream& operator<<(std::ostream& ostr, const NNet::Layer*& layer) {
	return layer->Write(ostr);
}