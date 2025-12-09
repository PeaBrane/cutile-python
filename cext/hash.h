/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once


#include <cstdint>
#include <type_traits>


// "Fx Hash" of the rustc fame
class Hasher {
public:
    void hash(uint64_t val) {
        uint64_t rot = (state_ << 5) | (state_ >> 59);
        state_ = (rot ^ val) * 0x517cc1b727220a95ull;
    }

    uint64_t get() const {
        return state_;
    }

private:
    uint64_t state_ = 0;
};


template <typename T, typename = void>
struct Hash {};


template <>
struct Hash<long long> {
    static void hash(long long x, Hasher& h) {
        h.hash(x);
    }
};


template <>
struct Hash<unsigned long> {
    static void hash(unsigned long x, Hasher& h) {
        h.hash(x);
    }
};


template <>
struct Hash<int> {
    static void hash(int x, Hasher& h) {
        h.hash(x);
    }
};


template <>
struct Hash<long> {
    static void hash(long x, Hasher& h) {
        h.hash(x);
    }
};

template <>
struct Hash<unsigned long long> {
    static void hash(unsigned long long x, Hasher& h) {
        h.hash(x);
    }
};


template <typename T>
struct Hash<T*> {
    static void hash(T* ptr, Hasher& h) {
        uint64_t x = reinterpret_cast<uintptr_t>(ptr);
        // Pointers are typically 8-byte aligned, so make sure we have enough entropy
        // in the low 3 bits
        x ^= x >> 3;
        return Hash<uint64_t>::hash(x, h);
    }
};


template <typename T>
struct Hash<T, typename std::enable_if_t<std::is_enum_v<T>> > {
    static void hash(T val, Hasher& h) {
        return Hash<uint64_t>::hash(static_cast<uint64_t>(val), h);
    }
};


