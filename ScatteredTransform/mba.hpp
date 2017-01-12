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

#include <boost/array.hpp>
#include <boost/multi_array.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/numeric.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/type_traits.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/function.hpp>

namespace mba {
namespace detail {

template <size_t N, size_t M>
struct power : boost::integral_constant<size_t, N * power<N, M-1>::value> {};

template <size_t N>
struct power<N, 0> : boost::integral_constant<size_t, 1> {};

/// N-dimensional grid iterator (nested loop with variable depth).
template <unsigned NDim>
class grid_iterator {
    public:
        typedef boost::array<size_t, NDim> index;

        explicit grid_iterator(const boost::array<size_t, NDim> &dims)
            : N(dims), idx(0)
        {
            boost::fill(i, 0);
            done = (i == N);
        }

        explicit grid_iterator(size_t dim) : idx(0) {
            boost::fill(N, dim);
            boost::fill(i, 0);
            done = (0 == dim);
        }

        size_t operator[](size_t d) const {
            return i[d];
        }

        const index& operator*() const {
            return i;
        }

        size_t position() const {
            return idx;
        }

        grid_iterator& operator++() {
            done = true;
			for(size_t d = 0; d<NDim; d++) {  // Grand Joldes: changed order
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
        index N, i;
        bool  done;
        size_t idx;
};

template <typename T, size_t N>
boost::array<T, N> operator+(boost::array<T, N> a, const boost::array<T, N> &b) {
    boost::transform(a, b, boost::begin(a), std::plus<T>());
    return a;
}

template <typename T, size_t N, typename C>
boost::array<T, N> operator-(boost::array<T, N> a, C b) {
    boost::transform(a, boost::begin(a), std::bind2nd(std::minus<T>(), b));
    return a;
}

template <typename T, size_t N, typename C>
boost::array<T, N> operator*(boost::array<T, N> a, C b) {
    boost::transform(a, boost::begin(a), std::bind2nd(std::multiplies<T>(), b));
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
bool boxed(const boost::array<T,N> &lo, const boost::array<T,N> &p, const boost::array<T,N> &hi) {
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
		typedef boost::array<size_t, NDim> index;
        typedef boost::array<double, NDim> point;
		typedef boost::multi_array<double, NDim> latticeType;

        template <class CooIter, class ValIter>
        control_lattice_dense(
                const point &coo_min, const point &coo_max, index grid_size,
                CooIter coo_begin, CooIter coo_end, ValIter val_begin, 
				boost::function<double(point)> initial = boost::function<double(point)>() 
                ) : cmin(coo_min), cmax(coo_max), grid(grid_size)
        {
            for(unsigned i = 0; i < NDim; ++i) {
                hinv[i] = (grid[i] - 1) / (cmax[i] - cmin[i]);
                cmin[i] -= 1 / hinv[i];
                grid[i] += 2;
            }
			phi.resize(grid);

			// Grand Joldes: init control grid values 
			if (initial)
			{
				point s;
				for(grid_iterator<NDim> d(grid); d; ++d) {
					for(unsigned k = 0; k < NDim; ++k) 
					{
						s[k] = cmin[k] + d[k]/hinv[k];
					}
					phi(*d) = initial(s);
				}
				residual(coo_begin, coo_end, val_begin); // substract values computed using current grid from interpolated values
            }

            latticeType delta(grid);
            latticeType omega(grid);
			latticeType initial_phi(grid);

            std::fill(delta.data(), delta.data() + delta.num_elements(), 0.0);
            std::fill(omega.data(), omega.data() + omega.num_elements(), 0.0);

            CooIter p = coo_begin;
            ValIter v = val_begin;

            for(; p != coo_end; ++p, ++v) {
                if (!boxed(coo_min, *p, coo_max)) continue;

                index i;
                point s;

                for(unsigned d = 0; d < NDim; ++d) {
                    double u = ((*p)[d] - cmin[d]) * hinv[d];
                    i[d] = (size_t)floor(u) - 1;  // Grand Joldes: added conversion to size_t
                    s[d] = u - floor(u);
                }

                boost::array< double, power<4, NDim>::value > w;
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
                    double dphi = (*v) * w1 / sum_w2;

                    index j = i + (*d);

                    delta(j) += w2 * dphi;
                    omega(j) += w2;
                }
            }

			// Grand Joldes: save initial grid
			if (initial) 
			{
				restore_values(coo_begin, coo_end, val_begin); // restore interpolated values
				initial_phi = phi;
			}

            std::transform(
                    delta.data(), delta.data() + delta.num_elements(),
                    omega.data(), phi.data(), safe_divide
                    );

			// Grand Joldes: handle initial grid
			if (initial) 
			{
				std::transform(
						phi.data(), phi.data() + phi.num_elements(),
						initial_phi.data(), phi.data(), std::plus<double>()
						);
			}
        }

        double operator()(const point &p) const {
            index i;
            point s;

			if (!boxed(cmin, p, cmax)) return 0; // Grand Joldes: check limits

            for(unsigned d = 0; d < NDim; ++d) {
                double u = (p[d] - cmin[d]) * hinv[d];
                i[d] = (size_t)floor(u) - 1;
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

		template <class CooIter, class ValIter>
        double residual(CooIter coo_begin, CooIter coo_end, ValIter val_begin) const {
            double res = 0.0;

            CooIter p = coo_begin;
            ValIter v = val_begin;

            for(; p != coo_end; ++p, ++v) {
				if (!boxed(cmin, *p, cmax)) continue;
                (*v) -= (*this)(*p);
                res = std::max(res, std::abs(*v));
            }

            return res;
        }

		// Grand Joldes: use this to restore interpolated values after a call to residual()
		template <class CooIter, class ValIter>
        void restore_values(CooIter coo_begin, CooIter coo_end, ValIter val_begin) const {
            CooIter p = coo_begin;
			ValIter v = val_begin;
            
            for(; p != coo_end; ++p, ++v) {
				if (!boxed(cmin, *p, cmax)) continue;
                (*v) += (*this)(*p);
            }
        }

        void report(std::ostream &os) const {
            os << "dense  [" << grid[0];
            for(unsigned i = 1; i < NDim; ++i)
                os << ", " << grid[i];
			os << "] (" << phi.num_elements() * sizeof(double) << " bytes)" << std::endl;

			// Grand Joldes: print coefficients
			os << "Coefficients: ";
			for(grid_iterator<NDim> i(grid); i; ++i) 
			{
                double f = phi(*i);
				os << f << "  ";
			}
			os << std::endl;
        }

        void append_refined(const control_lattice_dense &r) {
            static const boost::array<double, 5> s = {
                0.125, 0.500, 0.750, 0.500, 0.125
            };

            for(grid_iterator<NDim> i(r.grid); i; ++i) {
                double f = r.phi(*i);

                if (f == 0.0) continue;

                for(grid_iterator<NDim> d(5); d; ++d) {
                    index j;
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

		// Grand Joldes: access grid size
		index* getGridSize() { return &grid;};

		// Grand Joldes: access control lattice
		latticeType* getControlLattice() { return &phi;};
      
    private:
        point cmin, cmax, hinv;
        index grid;

        latticeType phi;

};

} // namespace detail

template <unsigned NDim>
class linear_approximation {
    public:
        typedef boost::array<double, NDim> point;
		typedef boost::array<double, NDim+1> arrayCoefficientsType;

        template <class CooIter, class ValIter>
        linear_approximation(CooIter coo_begin, CooIter coo_end, ValIter val_begin)
        {
            namespace ublas = boost::numeric::ublas;

            size_t n = std::distance(coo_begin, coo_end);

            if (n <= NDim) {
                // Not enough points to get a unique plane
                boost::fill(C, 0.0);
                C[NDim] = std::accumulate(val_begin, val_begin + n, 0.0) / n;
                return;
            }

            ublas::matrix<double> A(NDim+1, NDim+1); A.clear();
            ublas::vector<double> f(NDim+1);         f.clear();

            CooIter p = coo_begin;
            ValIter v = val_begin;

            double sum_val = 0.0;

            // Solve least-squares problem to get approximation with a plane.
            for(; p != coo_end; ++p, ++v, ++n) {
                boost::array<double, NDim+1> x;
                boost::copy(*p, boost::begin(x));
                x[NDim] = 1.0;

                for(unsigned i = 0; i <= NDim; ++i) {
                    for(unsigned j = 0; j <= NDim; ++j) {
                        A(i,j) += x[i] * x[j];
                    }
                    f(i) += x[i] * (*v);
                }

                sum_val += (*v);
            }

            ublas::permutation_matrix<size_t> pm(NDim+1);
            ublas::lu_factorize(A, pm);

            bool singular = false;
            for(unsigned i = 0; i <= NDim; ++i) {
                if (A(i,i) == 0.0) {
                    singular = true;
                    break;
                }
            }

            if (singular) {
                boost::fill(C, 0.0);
                C[NDim] = sum_val / n;
            } else {
                ublas::lu_substitute(A, pm, f);
                for(unsigned i = 0; i <= NDim; ++i) C[i] = f(i);
            }
        }

        double operator()(const point &p) const {
            double f = C[NDim];

            for(unsigned i = 0; i < NDim; ++i)
                f += C[i] * p[i];

            return f;
        }

		// Grand Joldes: print coefficients
		friend std::ostream& operator<<(std::ostream &os, const linear_approximation &La) {
            os << "Linear approximation coefficients: ";
			for(unsigned i = 0; i <= NDim; ++i)
            {
                os << La.C[i] << "  "; 
            }
			os << std::endl;
            return os;
        }

		// Grand Joldes: access coefficients
		arrayCoefficientsType *getCoefficients() {return &C;};
    private:
        arrayCoefficientsType C;
};

template <unsigned NDim>
class MBA {
    public:
        typedef boost::array<size_t, NDim> index;
        typedef boost::array<double, NDim> point;
		typedef boost::multi_array<double, NDim> latticeType;

        template <class CooIter, class ValIter>
        MBA(
                const point &coo_min, const point &coo_max, const index grid,
                CooIter coo_begin, CooIter coo_end, ValIter val_begin,
                unsigned max_levels = 8, double tol = 1e-8,
                boost::function<double(point)> initial = boost::function<double(point)>()
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
                const point &coo_min, const point &coo_max, const index grid,
                CooRange coo, ValRange val,
                unsigned max_levels = 8, double tol = 1e-8,
                boost::function<double(point)> initial = boost::function<double(point)>()
           )
        {
            init(
                    coo_min, coo_max, grid,
                    boost::begin(coo), boost::end(coo), boost::begin(val),
                    max_levels, tol, initial
                );
        }

        double operator()(const point &p) const {
            double f = (*dense_lattice_ptr)(p);
            return f;
        }

        friend std::ostream& operator<<(std::ostream &os, const MBA &h) {
            h.dense_lattice_ptr->report(os);
            os << std::endl;
            return os;
        }


		// Grand Joldes: access grid size
		index* getGridSize()
		{
			return dense_lattice_ptr->getGridSize();
		}

		// Grand Joldes: access control lattice
		latticeType* getControlLattice() 
		{
			return dense_lattice_ptr->getControlLattice();
		}

		// Grand Joldes: access level
		size_t getLevel() {return level;};

		// Grand Joldes: increase level
		template <class CooIter, class ValIter>
		void vIncreaseRefinementLevel(CooIter coo_begin, CooIter coo_end, ValIter val_begin, size_t newLevel)
		{
			using namespace mba::detail;
 
            const ptrdiff_t n = std::distance(coo_begin, coo_end);
            std::vector<double> val(val_begin, val_begin + n);

			dense_lattice_ptr->residual(coo_begin, coo_end, val.begin());

			// Refine control lattice.
            for(; (level < newLevel); ++level) {
                grid_size = grid_size * 2ul - 1ul;

				// create refined control lattice
                boost::shared_ptr<dense_lattice_type> f = boost::make_shared<dense_lattice_type>(
                        coo_min, coo_max, grid_size, coo_begin, coo_end, val.begin());

                f->residual(coo_begin, coo_end, val.begin());
                
                f->append_refined(*dense_lattice_ptr);
                dense_lattice_ptr.swap(f);
            }
		}

    private:
		typedef detail::control_lattice_dense<NDim> dense_lattice_type;
		size_t level;
		point coo_min, coo_max;
		index grid_size;

		boost::shared_ptr<dense_lattice_type> dense_lattice_ptr;
       
        template <class CooIter, class ValIter>
        void init(
                const point &cmin, const point &cmax, const index grid,
                CooIter coo_begin, CooIter coo_end, ValIter val_begin,
                unsigned max_levels, double tol,
                boost::function<double(point)> initial
                )
        {
            using namespace mba::detail;
			coo_min = cmin; 
			coo_max = cmax; 
			grid_size = grid;
            const ptrdiff_t n = std::distance(coo_begin, coo_end);
            std::vector<double> val(val_begin, val_begin + n);

			level = 1;
            // Create dense control lattice.
            dense_lattice_ptr = boost::make_shared<dense_lattice_type>(
                       cmin, cmax, grid_size, coo_begin, coo_end, val.begin(), initial);

            double res = dense_lattice_ptr->residual(coo_begin, coo_end, val.begin());

			if (res <= tol) return;
            
            for(; (level < max_levels) && (res > tol); ++level) {
                grid_size = grid_size * 2ul - 1ul;

				// create refined control lattice
                boost::shared_ptr<dense_lattice_type> f = boost::make_shared<dense_lattice_type>(
                        cmin, cmax, grid_size, coo_begin, coo_end, val.begin());

                res = f->residual(coo_begin, coo_end, val.begin());
                
                f->append_refined(*dense_lattice_ptr);
                dense_lattice_ptr.swap(f);
            }
        }
};

} // namespace mba

#endif
