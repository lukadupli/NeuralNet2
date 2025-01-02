#pragma once

#include <iostream>
#include <string>

namespace NNet {
	class Exception {
	public:
		Exception(const std::string& message) {
			std::cerr << "Program terminated.\n" << message;
		}
	};
}