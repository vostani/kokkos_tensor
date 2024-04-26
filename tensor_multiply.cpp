#include <pybind11/pybind11.h>
#include <Kokkos_Core.hpp>
#include <iostream>

namespace py = pybind11;

// function to initialize kokkos
void init(){
   Kokkos::initialize();
}

// function to finalize kokkos
void finalize(){
   Kokkos::finalize();
}

struct DeviceBuffer {
    float* float_ptr;
    void* void_ptr;
    unsigned long long ull_ptr;
    std::size_t size;
};

DeviceBuffer process_tensor(unsigned long long ptr, float* fptr, std::size_t size) {
    float* ptr_as_float = (float*) ptr;

    std::cout << "in process" << std::endl;

    std::cout << "ptr: " << ptr << std::endl;
    std::cout << "fptr: " << fptr << std::endl;
    std::cout << "(float*) ptr: " << ptr_as_float << std::endl;
    std::cout << "(uintptr_t) ptr_as_float: " << (uintptr_t) ptr_as_float << std::endl;
    std::cout << "(uintptr_t) fptr: " << (uintptr_t) fptr << std::endl;
    std::cout << "ptr - fptr: " << (uintptr_t) fptr - (uintptr_t) ptr << std::endl;

    // Create an unmanaged Kokkos view from the raw pointer
    Kokkos::View<float*, Kokkos::DefaultExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> data_view(ptr_as_float, size);

    std::cerr << "before parallel \n";
    // Use Kokkos to perform some operations (e.g., scaling)
    Kokkos::parallel_for("scale_data", size-1, [=] __device__(const int i) {
        printf("thread %d %g \n", i, data_view(i));
        data_view(i) *= 2.0f;
        printf("thread %d %g \n", i, data_view(i));
        __syncthreads();
    });
    cudaDeviceSynchronize();
    std::cout.flush();
    std::cerr.flush();

    std::cerr << "after parallel \n";

    //Kokkos::fence();  // Ensure Kokkos operations are completed
 
    std::cout << "dv_ptr: " << data_view.data() << std::endl;
    std::cout << "(uintptr_t) dv_ptr: " << (uintptr_t) data_view.data() << std::endl;

    return { data_view.data(), data_view.data(), (unsigned long long) data_view.data(), data_view.size() };
}

PYBIND11_MODULE(tenmul, m) {
    // call the init and finalize functions
    m.def("init", &init);
    m.def("finalize", &finalize);

    m.def("process_tensor", &process_tensor, "Process a tensor using Kokkos", py::return_value_policy::take_ownership);
    py::class_<DeviceBuffer>(m, "DeviceBuffer")
        .def_readonly("float_ptr", &DeviceBuffer::float_ptr)
        .def_readonly("void_ptr", &DeviceBuffer::void_ptr)
        .def_readonly("ull_ptr", &DeviceBuffer::ull_ptr)
        .def_readonly("size", &DeviceBuffer::size);
}
