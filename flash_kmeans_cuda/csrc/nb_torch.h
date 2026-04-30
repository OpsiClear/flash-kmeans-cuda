#pragma once

// nanobind <-> torch::Tensor bridge.
//
// nanobind ships no caster for at::Tensor. We supply a minimal one that
// unwraps a Python torch.Tensor via ``THPVariable_Unpack`` (returns a borrowed
// at::Tensor reference owned by the Python object) and re-wraps via
// ``THPVariable_Wrap`` (returns a new Python reference).
//
// For optional<at::Tensor>, we accept ``None`` -> empty optional, otherwise
// dispatch to the at::Tensor caster.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

// Windows.h (pulled in via Python.h on Windows) defines `small` as `char`,
// which collides with PyTorch headers that use `small` as a parameter name
// (e.g. c10/cuda/CUDACachingAllocator.h). Undef the offenders before pulling
// in torch.
#ifdef small
  #undef small
#endif
#ifdef min
  #undef min
#endif
#ifdef max
  #undef max
#endif

#include <torch/extension.h>
#include <torch/csrc/autograd/python_variable.h>
#include <c10/util/Optional.h>

namespace nb = nanobind;

namespace nanobind { namespace detail {

template <>
struct type_caster<at::Tensor> {
  NB_TYPE_CASTER(at::Tensor, const_name("torch.Tensor"))

  bool from_python(handle src, uint8_t /*flags*/, cleanup_list* /*cleanup*/) noexcept {
    PyObject* obj = src.ptr();
    if (!obj || obj == Py_None) return false;
    if (!THPVariable_Check(obj)) return false;
    new (&value) at::Tensor(THPVariable_Unpack(obj));
    return true;
  }

  static handle from_cpp(at::Tensor v, rv_policy /*policy*/, cleanup_list* /*cleanup*/) noexcept {
    return THPVariable_Wrap(std::move(v));
  }
};

// c10::optional<at::Tensor> caster — None maps to nullopt.
template <>
struct type_caster<c10::optional<at::Tensor>> {
  NB_TYPE_CASTER(c10::optional<at::Tensor>, const_name("Optional[torch.Tensor]"))

  bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept {
    PyObject* obj = src.ptr();
    if (!obj || obj == Py_None) {
      new (&value) c10::optional<at::Tensor>(c10::nullopt);
      return true;
    }
    type_caster<at::Tensor> inner;
    if (!inner.from_python(src, flags, cleanup)) return false;
    new (&value) c10::optional<at::Tensor>(std::move(inner.value));
    return true;
  }

  static handle from_cpp(c10::optional<at::Tensor> v, rv_policy policy, cleanup_list* cleanup) noexcept {
    if (!v.has_value()) {
      Py_INCREF(Py_None);
      return Py_None;
    }
    return type_caster<at::Tensor>::from_cpp(std::move(*v), policy, cleanup);
  }
};

}}  // namespace nanobind::detail
