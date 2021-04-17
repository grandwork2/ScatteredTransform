#ifndef MBA_MBA_HPP
#define MBA_MBA_HPP

/*
The MIT License

Copyright (c) 2015 Denis Demidov <dennis.demidov@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * \file   mba/mba.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Multilevel B-spline interpolation.
 * 
 * \changes Grand Joldes <grandwork2@yahoo.com>: 
 *			Removed the sparse control lattice and initial approximation.
 *			Removed the parent class of control_lattice_dense.
 *			Transferred the initial approximation values into the dense control lattice coefficients.
 *			Use absolute tolerance as termination criterion.
 *			Added access functions for the control lattice, grid size and linear approximation coefficients.
 *			Added function to increase refinement level
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>
#include <functional>
#include <type_traits>
#include <utility>
#include <iterator>
#include <stdexcept>

#include <cmath>
#include <cassert>

namespace mba {

using std::ptrdiff_t;
using std::size_t;

template <int N>
using point = std::array<double, N>;

template <int N>
using index = std::array<size_t, N>;

namespace detail {

// Compile-time N^M
template <size_t N, size_t M>
struct power : std::integral_constant<size_t, N * power<N, M-1>::value> {};

template <size_t N>
struct power<N, 0> : std::integral_constant<size_t, 1> {};

// N-dimensional dense matrix
template <class T, int N>
class multi_array {
    static_assert(N > 0, "Wrong number of dimensions");

    public:
        multi_array(): stride(), buf() {}

        multi_array(index<N> n) {
            init(n);
        }

        void resize(index<N> n) {
            init(n);
        }

        size_t size() const {
            return buf.size();
        }

        T operator()(index<N> i) const {
            return buf[idx(i)];
        }

        T& operator()(index<N> i) {
            return buf[idx(i)];
        }

        T operator[](size_t i) const {
            return buf[i];
        }

        T& operator[](size_t i) {
            return buf[i];
        }

        const T* data() const {
            return buf.data();
        }

        T* data() {
            return buf.data();
        }
    private:
        std::array<int, N> stride;
        std::vector<T>  buf;

        void init(index<N> n) {
            size_t s = 1;

            for(int d = N-1; d >= 0; --d) {
                stride[d] = s;
                s *= n[d];
            }

            buf.resize(s);
        }

        size_t idx(index<N> i) const {
            size_t p = 0;
            for(int d = 0; d < N; ++d)
                p += stride[d] * i[d];
            return p;
        }
};

/// N-dimensional grid iterator (nested loop with variable depth).
template <unsigned NDim>
class grid_iterator {
    public:
        explicit grid_iterator(const index<NDim> &dims)
            : N(dims), i{0}, done(i == N), idx(0) { }

        explicit grid_iterator(size_t dim) : i{0}, idx(0), done(dim == 0) {
            for(auto &v : N)  v = dim;
        }

        size_t operator[](size_t d) const {
            return i[d];
        }

        const index<NDim>& operator*() const {
            return i;
        }

        size_t position() const {
            return idx;
        }

        grid_iterator& operator++() {
            done = true;
            for (size_t d = 0; d < NDim; d++) {  // Grand Joldes: changed order
                if (++i[d] < N[d]) {
                    done = false;
                    break;
                }
                i[d] = 0;
            }

            ++idx;

            return *this;
        }

        operator bool() const { return !done; }

    private:
        index<NDim> N, i;
        bool  done;
        size_t idx;
};

template <typename T, size_t N>
std::array<T, N> operator+(std::array<T, N> a, const std::array<T, N> &b) {
    for(size_t i = 0; i < N; ++i) a[i] += b[i];
    return a;
}

template <typename T, size_t N, typename C>
std::array<T, N> operator-(std::array<T, N> a, C b) {
    for(auto &v : a) v -= b;
    return a;
}

template <typename T, size_t N, typename C>
std::array<T, N> operator*(std::array<T, N> a, C b) {
    for(auto &v : a) v *= b;
    return a;
}

// Value of k-th B-Spline basic function at t.
inline double Bspline(size_t k, double t) {
    assert(0 <= t && t < 1);
    assert(k < 4);

    switch (k) {
        case 0:
            return (t * (t * (-t + 3) - 3) + 1) / 6;
        case 1:
            return (t * t * (3 * t - 6) + 4) / 6;
        case 2:
            return (t * (t * (-3 * t + 3) + 3) + 1) / 6;
        case 3:
            return t * t * t / 6;
        default:
            return 0;
    }
}

// Checks if p is between lo and hi
template <typename T, size_t N>
bool boxed(const std::array<T,N> &lo, const std::array<T,N> &p, const std::array<T,N> &hi) {
    for(unsigned i = 0; i < N; ++i) {
        if (p[i] < lo[i] || p[i] > hi[i]) return false;
    }
    return true;
}

inline double safe_divide(double a, double b) {
    return b == 0.0 ? 0.0 : a / b;
}


template <unsigned NDim>
class control_lattice_dense {
    public:
        template <class CooIter, class ValIter>
        control_lattice_dense(
                const point<NDim> &coo_min, const point<NDim> &coo_max, index<NDim> grid_size,
                CooIter coo_begin, CooIter coo_end, ValIter val_begin,
                std::function<double(point<NDim>)> initial = std::function<double(point<NDim>)>()
                ) : cmin(coo_min), cmax(coo_max), grid(grid_size), residual( 0.0 )
        {
            static_assert(
                    std::is_same<
                        typename std::iterator_traits<CooIter>::iterator_category,
                        std::random_access_iterator_tag
                        >::value,
                    "CooIter should be a random access iterator");
            static_assert(
                    std::is_same<
                        typename std::iterator_traits<ValIter>::iterator_category,
                        std::random_access_iterator_tag
                        >::value,
                    "ValIter should be a random access iterator");

            for(unsigned i = 0; i < NDim; ++i) {
                hinv[i] = (grid[i] - 1) / (cmax[i] - cmin[i]);
                cmin[i] -= 1 / hinv[i];
                grid[i] += 2;
            }
            phi.resize(grid);
            std::fill(phi.data(), phi.data() + phi.size(), 0.0);

            // Grand Joldes: init control grid values 
            if (initial)
            {
                point<NDim> s;
                for (grid_iterator<NDim> d(grid); d; ++d) {
                    for (unsigned k = 0; k < NDim; ++k)
                    {
                        s[k] = cmin[k] + d[k] / hinv[k];
                    }
                    phi(*d) = initial(s);
                }
                update_residual(coo_begin, coo_end, val_begin); // substract values computed using current grid from interpolated values
            }

            multi_array<double, NDim> delta(grid);
            multi_array<double, NDim> omega(grid);

            std::fill(delta.data(), delta.data() + delta.size(), 0.0);
            std::fill(omega.data(), omega.data() + omega.size(), 0.0);

            ptrdiff_t n = std::distance(coo_begin, coo_end);
            size_t    m = phi.size();

            for(ptrdiff_t l = 0; l < n; ++l) {
                auto p = coo_begin[l];
                auto v = val_begin[l];

                if (!boxed(coo_min, p, coo_max)) continue;

                index<NDim> i;
                point<NDim> s;

                for(unsigned d = 0; d < NDim; ++d) {
                    double u = (p[d] - cmin[d]) * hinv[d];
                    i[d] = (size_t)floor(u) - 1;
                    s[d] = u - floor(u);
                }

                std::array< double, power<4, NDim>::value > w;
                double sum_w2 = 0.0;

                for(grid_iterator<NDim> d(4); d; ++d) {
                    double prod = 1.0;
                    for(unsigned k = 0; k < NDim; ++k) prod *= Bspline(d[k], s[k]);

                    w[d.position()] = prod;
                    sum_w2 += prod * prod;
                }

                for(grid_iterator<NDim> d(4); d; ++d) {
                    double w1  = w[d.position()];
                    double w2  = w1 * w1;
                    double phi = v * w1 / sum_w2;

                    index<NDim> j = i + (*d);

                    delta(j) += w2 * phi;
                    omega(j) += w2;
                }
            }

            // Grand Joldes: handle initial grid
            if (initial)
            {
                restore_values(coo_begin, coo_end, val_begin); // restore interpolated values
                for (ptrdiff_t i = 0; i < m; ++i) {
                    phi[i] += safe_divide(delta[i], omega[i]);
                }
            }
            else
            {
                for (ptrdiff_t i = 0; i < m; ++i) {
                    phi[i] = safe_divide(delta[i], omega[i]);
                }
            }
            residual = update_residual(coo_begin, coo_end, val_begin); // substract values computed using current grid from interpolated values
        }

        double operator()(const point<NDim> &p) const {
            index<NDim> i;
            point<NDim> s;

            for(unsigned d = 0; d < NDim; ++d) {
                double u = (p[d] - cmin[d]) * hinv[d];
                i[d] = floor(u) - 1;
                s[d] = u - floor(u);
            }

            double f = 0;

            for(grid_iterator<NDim> d(4); d; ++d) {
                double w = 1.0;
                for(unsigned k = 0; k < NDim; ++k) w *= Bspline(d[k], s[k]);

                f += w * phi(i + (*d));
            }

            return f;
        }

        void report(std::ostream &os) const {
            std::ios_base::fmtflags ff(os.flags());
            auto fp = os.precision();

            os << "dense  [" << grid[0];
            for(unsigned i = 1; i < NDim; ++i)
                os << ", " << grid[i];
            os << "] (" << phi.size() * sizeof(double) << " bytes)";

            os.flags(ff);
            os.precision(fp);
        }

        void append_refined(const control_lattice_dense &r) {
            static const std::array<double, 5> s = {
                0.125, 0.500, 0.750, 0.500, 0.125
            };

            for(grid_iterator<NDim> i(r.grid); i; ++i) {
                double f = r.phi(*i);

                if (f == 0.0) continue;

                for(grid_iterator<NDim> d(5); d; ++d) {
                    index<NDim> j;
                    bool skip = false;
                    for(unsigned k = 0; k < NDim; ++k) {
                        j[k] = 2 * i[k] + d[k] - 3;
                        if (j[k] >= grid[k]) {
                            skip = true;
                            break;
                        }
                    }

                    if (skip) continue;

                    double c = 1.0;
                    for(unsigned k = 0; k < NDim; ++k) c *= s[d[k]];

                    phi(j) += f * c;
                }
            }
        }

        double fill_ratio() const {
            size_t total    = phi.size();
            size_t nonzeros = total - std::count(phi.data(), phi.data() + total, 0.0);

            return static_cast<double>(nonzeros) / total;
        }

        // Grand Joldes: access grid size
        index<NDim>* getGridSize() { return &grid; };

        // Grand Joldes: access control lattice
        multi_array<double, NDim>* getControlLattice() { return &phi; };

        double getResidual() { return residual; };

    private:
        template <class CooIter, class ValIter>
        double update_residual(CooIter coo_begin, CooIter coo_end, ValIter val_begin) const {
            double res = 0.0;

            CooIter p = coo_begin;
            ValIter v = val_begin;

            for (; p != coo_end; ++p, ++v) {
                if (!boxed(cmin, *p, cmax)) continue;
                (*v) -= (*this)(*p);
                res = std::max(res, std::abs(*v));
            }
            return res;
        }

        // Grand Joldes: use this to restore interpolated values after a call to update_residual()
        template <class CooIter, class ValIter>
        void restore_values(CooIter coo_begin, CooIter coo_end, ValIter val_begin) const {
            CooIter p = coo_begin;
            ValIter v = val_begin;

            for (; p != coo_end; ++p, ++v) {
                if (!boxed(cmin, *p, cmax)) continue;
                (*v) += (*this)(*p);
            }
        }

        point<NDim> cmin, cmax, hinv;
        index<NDim> grid;
        double residual;
        multi_array<double, NDim> phi;
};

} // namespace detail

template <unsigned NDim>
class linear_approximation {
    public:
        template <class CooIter, class ValIter>
        linear_approximation(CooIter coo_begin, CooIter coo_end, ValIter val_begin)
        {
            size_t n = std::distance(coo_begin, coo_end);
            for(auto &v : C) v = 0.0;

            if (n <= NDim) {
                // Not enough points to get a unique plane
                C[NDim] = std::accumulate(val_begin, val_begin + n, 0.0) / n;
                return;
            }

            double A[NDim + 1][NDim + 1] = {0};

            CooIter p = coo_begin;
            ValIter v = val_begin;

            double sum_val = 0.0;

            // Solve least-squares problem to get approximation with a plane.
            for(; p != coo_end; ++p, ++v, ++n) {
                double x[NDim+1];
                for(unsigned i = 0; i < NDim; ++i) x[i] = (*p)[i];
                x[NDim] = 1.0;

                for(unsigned i = 0; i <= NDim; ++i) {
                    for(unsigned j = 0; j <= NDim; ++j) {
                        A[i][j] += x[i] * x[j];
                    }
                    C[i] += x[i] * (*v);
                }

                sum_val += (*v);
            }

            // Perform LU-factorization of A in-place
            for(unsigned k = 0; k <= NDim; ++k) {
                if (A[k][k] == 0)
                {
                    // singular matrix
                    for (unsigned i = 0; i < NDim; ++i) C[i] = 0.0;
                    C[NDim] = sum_val / n;
                    return;
                }

                double d = 1 / A[k][k];
                for(unsigned i = k+1; i <= NDim; ++i) {
                    A[i][k] *= d;
                    for(unsigned j = k+1; j <= NDim; ++j)
                        A[i][j] -= A[i][k] * A[k][j];
                }
                A[k][k] = d;
            }

            for (unsigned i = 0; i <= NDim; ++i)
            {
                if (A[i][i] == 0.0)
                {
                    // singular matrix
                    for (unsigned i = 0; i < NDim; ++i) C[i] = 0.0;
                    C[NDim] = sum_val / n;
                    return;
                }
            }

            // Lower triangular solve:
            for(unsigned i = 0; i <= NDim; ++i) {
                for(unsigned j = 0; j < i; ++j)
                    C[i] -= A[i][j] * C[j];
            }

            // Upper triangular solve:
            for(unsigned i = NDim+1; i --> 0; ) {
                for(unsigned j = i+1; j <= NDim; ++j)
                    C[i] -= A[i][j] * C[j];
                C[i] *= A[i][i];
            }
        }

        double operator()(const point<NDim> &p) const {
            double f = C[NDim];

            for(unsigned i = 0; i < NDim; ++i)
                f += C[i] * p[i];

            return f;
        }

        // Grand Joldes: print coefficients
        friend std::ostream& operator<<(std::ostream& os, const linear_approximation& La) {
            os << "Linear approximation coefficients: ";
            for (unsigned i = 0; i <= NDim; ++i)
            {
                os << La.C[i] << "  ";
            }
            os << std::endl;
            return os;
        }

    private:
        std::array<double, NDim+1> C;
};

using namespace mba::detail;

template <unsigned NDim>
class MBA {
    public:
        typedef multi_array<double, NDim> latticeType;
        typedef index<NDim> index_type;
        typedef point<NDim> point_type;

        template <class CooIter, class ValIter>
        MBA(
                const point<NDim> &coo_min, const point<NDim> &coo_max, index<NDim> grid,
                CooIter coo_begin, CooIter coo_end, ValIter val_begin,
                unsigned max_levels = 8, double tol = 1e-8,
                std::function<double(point<NDim>)> initial = std::function<double(point<NDim>)>()
           )
        {
            init(
                    coo_min, coo_max, grid,
                    coo_begin, coo_end, val_begin,
                    max_levels, tol, initial
                );
        }

        template <class CooRange, class ValRange>
        MBA(
                const point<NDim> &coo_min, const point<NDim> &coo_max, index<NDim> grid,
                CooRange coo, ValRange val,
                unsigned max_levels = 8, double tol = 1e-8,
                std::function<double(point<NDim>)> initial = std::function<double(point<NDim>)>()
           )
        {
            init(
                    coo_min, coo_max, grid,
                    std::begin(coo), std::end(coo), std::begin(val),
                    max_levels, tol, initial
                );
        }

        double operator()(const point<NDim> &p) const {
            double f = (*dense_lattice_ptr)(p);
            return f;
        }

        friend std::ostream& operator<<(std::ostream &os, const MBA &h) {
            h.dense_lattice_ptr->report(os);
            os << std::endl;
            return os;
        }

        // Grand Joldes: access grid size
        index_type* getGridSize() const
        {
            return dense_lattice_ptr->getGridSize();
        }

        // Grand Joldes: access control lattice
        latticeType* getControlLattice() const
        {
            return dense_lattice_ptr->getControlLattice();
        }

        // Grand Joldes: access level
        size_t getLevel() const { return m_level; };

        // Grand Joldes: access residual
        double getResidual() { return dense_lattice_ptr->getResidual(); };

        // Grand Joldes: increase refinement level
        template <class CooIter, class ValIter>
        void vIncreaseRefinementLevel(CooIter coo_begin, CooIter coo_end, ValIter val_begin, size_t newLevel)
        {
            // Refine control lattice.
            for (; (m_level < newLevel); ++m_level) {
                refine(coo_begin, coo_end, val_begin);
            }
        }
    private:
        typedef detail::control_lattice_dense<NDim>  dense_lattice_type;
        size_t m_level;
        point<NDim> coo_min, coo_max;
        index<NDim> grid_size;

        std::shared_ptr<dense_lattice_type> dense_lattice_ptr;

        template <class CooIter, class ValIter>
        void init(
                const point<NDim> &cmin, const point<NDim> &cmax, index<NDim> grid,
                CooIter coo_begin, CooIter coo_end, ValIter val_begin,
                unsigned max_levels, double tol,
                std::function<double(point<NDim>)> initial
                )
        {
            coo_min = cmin;
            coo_max = cmax;
            grid_size = grid;
            const ptrdiff_t n = std::distance(coo_begin, coo_end);
            std::vector<double> val(val_begin, val_begin + n);

            m_level = 1;
            // Create dense control lattice.
            dense_lattice_ptr = std::make_shared<dense_lattice_type>(
                cmin, cmax, grid_size, coo_begin, coo_end, val.begin(), initial);
            double residual = getResidual();
            if (residual <= tol) return;
            for (; (m_level < max_levels) && (residual > tol); ++m_level) {
                refine(coo_begin, coo_end, val.begin());
                residual = getResidual();
            }
        }

        template <class CooIter, class ValIter>
        void refine(CooIter coo_begin, CooIter coo_end, ValIter val_begin)
        {
            grid_size = grid_size * 2ul - 1ul;
            // create refined control lattice
            std::shared_ptr<dense_lattice_type> f = std::make_shared<dense_lattice_type>(
                coo_min, coo_max, grid_size, coo_begin, coo_end, val_begin);
            f->append_refined(*dense_lattice_ptr);
            dense_lattice_ptr.swap(f);
        }
};

} // namespace mba

#endif
