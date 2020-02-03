#ifndef LINEAR_SOLVERS_INCLUDE
#define LINEAR_SOLVERS_INCLUDE

#include "SparseMatrixInterface.h"

inline double SquareNorm(const double* values, int dim) {
    double norm2 = 0;
    for (int i = 0; i < dim; i++) norm2 += values[i] * values[i];
    return norm2;
}
inline double SquareNorm(const float* values, int dim) {
    double norm2 = 0;
    for (int i = 0; i < dim; i++) norm2 += values[i] * values[i];
    return norm2;
}
template <class Type>
inline double SquareNorm(const Type* values, int dim) {
    double norm2 = 0;
    for (int i = 0; i < dim; i++) norm2 += values[dim].squareNorm();
    return norm2;
}

inline double SquareDifference(const double* values1,
                               const double* values2,
                               int dim) {
    double norm2 = 0;
    for (int i = 0; i < dim; i++)
        norm2 += (values1[i] - values2[i]) * (values1[i] - values2[i]);
    return norm2;
}
inline double SquareDifference(const float* values1,
                               const float* values2,
                               int dim) {
    double norm2 = 0;
    for (int i = 0; i < dim; i++)
        norm2 += (values1[i] - values2[i]) * (values1[i] - values2[i]);
    return norm2;
}
template <class Type>
inline double SquareDifference(const Type* values1,
                               const Type* values2,
                               int dim) {
    double norm2 = 0;
    for (int i = 0; i < dim; i++)
        norm2 += (values1[dim] - values2[dim]).squareNorm();
    return norm2;
}

// This is the conjugate gradients solver.
// The assumption is that the class SPDOperator defines a method operator()(
// const Real* , Real* ) which corresponds to applying a symmetric
// positive-definite operator.
template <class Real>
struct CGScratch {
    Real *r, *d, *q;
    CGScratch(void) : r(NULL), d(NULL), q(NULL), _dim(0) { ; }
    CGScratch(int dim) : r(NULL), d(NULL), q(NULL) { resize(dim); }
    ~CGScratch(void) { resize(0); }
    void resize(int dim) {
        if (dim != _dim) {
            if (r) delete[] r;
            r = NULL;
            if (d) delete[] d;
            d = NULL;
            if (q) delete[] q;
            q = NULL;
            if (dim) r = new Real[dim], d = new Real[dim], q = new Real[dim];
            _dim = dim;
        }
    }

protected:
    int _dim;
};
template <class Real>
struct PreconditionedCGScratch : public CGScratch<Real> {
    Real* s;
    PreconditionedCGScratch(void) : CGScratch<Real>(), s(NULL) { ; }
    PreconditionedCGScratch(int dim) : CGScratch<Real>() { resize(dim); }
    ~PreconditionedCGScratch(void) { resize(0); }
    void resize(int dim) {
        if (dim != CGScratch<Real>::_dim) {
            if (s) delete[] s;
            s = NULL;
            if (dim) s = new Real[dim];
        }
        CGScratch<Real>::resize(dim);
    }
};
template <class Real>
struct DiagonalPreconditioner {
    Real* iDiagonal;
    DiagonalPreconditioner(void) : iDiagonal(NULL), _dim(0) { ; }
    ~DiagonalPreconditioner(void) {
        if (iDiagonal) delete[] iDiagonal;
        iDiagonal = NULL;
    }
    template <class MatrixRowIterator>
    void set(const SparseMatrixInterface<Real, MatrixRowIterator>& M) {
        if (_dim != M.rows()) {
            _dim = (int)M.rows();
            if (iDiagonal) delete[] iDiagonal, iDiagonal = NULL;
            if (_dim > 0) iDiagonal = new Real[_dim];
        }
        memset(iDiagonal, 0, sizeof(Real) * _dim);
#pragma omp parallel for
        for (int i = 0; i < M.rows(); i++) {
            for (MatrixRowIterator iter = M.begin(i); iter != M.end(i); iter++)
                if (iter->N == i) iDiagonal[i] += iter->Value;
            iDiagonal[i] = (Real)1. / iDiagonal[i];
        }
    }
    void operator()(const Real* in, Real* out) const {
#pragma omp parallel for
        for (int i = 0; i < _dim; i++) out[i] = in[i] * iDiagonal[i];
    }

protected:
    int _dim;
};

template <class Real, class SPDOperator>
int SolveCG(SPDOperator& L,
            int iters,
            int dim,
            const Real* b,
            Real* x,
            CGScratch<Real>* scratch = NULL,
            double eps = 1e-8,
            int threads = 1,
            bool verbose = false) {
    eps *= eps;
    Real *r, *d, *q;
    if (scratch)
        r = scratch->r, d = scratch->d, q = scratch->q;
    else
        r = new Real[dim], d = new Real[dim], q = new Real[dim];
    memset(r, 0, sizeof(Real) * dim), memset(d, 0, sizeof(Real) * dim),
            memset(q, 0, sizeof(Real) * dim);
    double delta_new = 0, delta_0;

    L(x, r);
#pragma omp parallel for reduction(+ : delta_new)
    for (int i = 0; i < dim; i++)
        d[i] = r[i] = b[i] - r[i], delta_new += r[i] * r[i];

    delta_0 = delta_new;
    if (delta_new < eps) {
        if (!scratch) delete[] r, delete[] d, delete[] q;
        return 0;
    }

    int ii;
    for (ii = 0; ii < iters && delta_new > eps * delta_0; ii++) {
        L(d, q);
        double dDotQ = 0;
#pragma omp parallel for reduction(+ : dDotQ)
        for (int i = 0; i < dim; i++) dDotQ += d[i] * q[i];
        Real alpha = Real(delta_new / dDotQ);

        double delta_old = delta_new;
        delta_new = 0;

        const int RESET_COUNT = 50;
        if ((ii % RESET_COUNT) == (RESET_COUNT - 1)) {
#pragma omp parallel for
            for (int i = 0; i < dim; i++) x[i] += d[i] * alpha;
            L(x, r);
#pragma omp parallel for reduction(+ : delta_new)
            for (int i = 0; i < dim; i++)
                r[i] = b[i] - r[i], delta_new += r[i] * r[i];
        } else
#pragma omp parallel for reduction(+ : delta_new)
            for (int i = 0; i < dim; i++)
                r[i] -= q[i] * alpha, delta_new += r[i] * r[i],
                        x[i] += d[i] * alpha;

        Real beta = Real(delta_new / delta_old);
#pragma omp parallel for
        for (int i = 0; i < dim; i++) d[i] = r[i] + d[i] * beta;
    }
    if (verbose) {
        L(x, r);
#pragma omp parallel for
        for (int i = 0; i < dim; i++) r[i] -= b[i];
        printf("CG: %d %g -> %g\n", ii, SquareNorm(b, dim), SquareNorm(r, dim));
    }
    if (!scratch) delete[] r, delete[] d, delete[] q;
    return ii;
}
template <class Real, class SPDOperator, class SPDPreconditioner>
int SolvePreconditionedCG(SPDOperator& L,
                          SPDPreconditioner& Pinverse,
                          int iters,
                          int dim,
                          const Real* b,
                          Real* x,
                          PreconditionedCGScratch<Real>* scratch = NULL,
                          double eps = 1e-8,
                          int threads = 1,
                          bool verbose = false) {
    eps *= eps;
    Real *r, *d, *q, *s;
    if (scratch)
        r = scratch->r, d = scratch->d, q = scratch->q, s = scratch->s;
    else
        r = new Real[dim], d = new Real[dim], q = new Real[dim],
        s = new Real[dim];
    memset(r, 0, sizeof(Real) * dim), memset(d, 0, sizeof(Real) * dim),
            memset(q, 0, sizeof(Real) * dim), memset(s, 0, sizeof(Real) * dim);
    double delta_new = 0, delta_0;

    L(x, r);
#pragma omp parallel for
    for (int i = 0; i < dim; i++) r[i] = b[i] - r[i];
    Pinverse(r, d);
#pragma omp parallel for reduction(+ : delta_new)
    for (int i = 0; i < dim; i++) delta_new += r[i] * d[i];

    delta_0 = delta_new;
    if (delta_new < eps) {
        if (!scratch) delete[] r, delete[] d, delete[] q;
        return 0;
    }
    int ii;
    for (ii = 0; ii < iters && delta_new > eps * delta_0; ii++) {
        L(d, q);
        double dDotQ = 0;
#pragma omp parallel for reduction(+ : dDotQ)
        for (int i = 0; i < dim; i++) dDotQ += d[i] * q[i];
        Real alpha = Real(delta_new / dDotQ);

        const int RESET_COUNT = 50;
#pragma omp parallel for
        for (int i = 0; i < dim; i++) x[i] += d[i] * alpha;
        if ((ii % RESET_COUNT) == (RESET_COUNT - 1)) {
            L(x, r);
#pragma omp parallel for
            for (int i = 0; i < dim; i++) r[i] = b[i] - r[i];
        } else
#pragma omp parallel for reduction(+ : delta_new)
            for (int i = 0; i < dim; i++) r[i] -= q[i] * alpha;
        Pinverse(r, s);

        double delta_old = delta_new;
        delta_new = 0;
#pragma omp parallel for reduction(+ : delta_new)
        for (int i = 0; i < dim; i++) delta_new += r[i] * s[i];

        Real beta = Real(delta_new / delta_old);
#pragma omp parallel for
        for (int i = 0; i < dim; i++) d[i] = s[i] + d[i] * beta;
    }
    if (verbose) {
        L(x, r);
#pragma omp parallel for
        for (int i = 0; i < dim; i++) r[i] -= b[i];
        printf("PCCG: %d %g -> %g\n", ii, SquareNorm(b, dim),
               SquareNorm(r, dim));
    }
    if (!scratch) delete[] r, delete[] d, delete[] q, delete[] s;
    return ii;
}

#ifdef USE_EIGEN
#define STORE_EIGEN_MATRIX
#ifdef EIGEN_USE_MKL_ALL
#include <Eigen/PardisoSupport>
#else  // !EIGEN_USE_MKL_ALL
#include <Eigen/Sparse>
#endif  // EIGEN_USE_MKL_ALL

template <class Real, class MatrixRowIterator>
struct EigenSolver {
    virtual void update(
            const SparseMatrixInterface<Real, MatrixRowIterator>& M) = 0;
    virtual void solve(ConstPointer(Real) b, Pointer(Real) x) = 0;
    virtual size_t dimension(void) const = 0;
};

template <class Real, class MatrixRowIterator>
class EigenSolverCholeskyLLt : public EigenSolver<Real, MatrixRowIterator> {
#ifdef EIGEN_USE_MKL_ALL
    typedef Eigen::PardisoLLT<Eigen::SparseMatrix<double>> Eigen_Solver;
    typedef Eigen::VectorXd Eigen_Vector;
#else   // !EIGEN_USE_MKL_ALL
    typedef Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> Eigen_Solver;
    typedef Eigen::VectorXd Eigen_Vector;
#endif  // EIGEN_USE_MKL_ALL
    Eigen_Solver _solver;
    Eigen_Vector _eigenB;
#ifdef STORE_EIGEN_MATRIX
    Eigen::SparseMatrix<double> _eigenM;
#endif  // STORE_EIGEN_MATRIX
public:
    EigenSolverCholeskyLLt(
            const SparseMatrixInterface<Real, MatrixRowIterator>& M,
            bool analyzeOnly = false) {
#ifdef STORE_EIGEN_MATRIX
        _eigenM.resize(int(M.rows()), int(M.rows()));
#else   // !STORE_EIGEN_MATRIX
        Eigen::SparseMatrix<double> eigenM(int(M.rows()), int(M.rows()));
#endif  // STORE_EIGEN_MATRIX
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(M.entries());
        for (int i = 0; i < M.rows(); i++)
            for (MatrixRowIterator iter = M.begin(i); iter != M.end(i); iter++)
                triplets.push_back(
                        Eigen::Triplet<double>(i, iter->N, iter->Value));
#ifdef STORE_EIGEN_MATRIX
        _eigenM.setFromTriplets(triplets.begin(), triplets.end());
        _solver.analyzePattern(_eigenM);
#else   // !STORE_EIGEN_MATRIX
        eigenM.setFromTriplets(triplets.begin(), triplets.end());
        _solver.analyzePattern(eigenM);
#endif  // STORE_EIGEN_MATRIX
        if (!analyzeOnly) {
#ifdef STORE_EIGEN_MATRIX
            _solver.factorize(_eigenM);
#else   // !STORE_EIGEN_MATRIX
            _solver.factorize(eigenM);
#endif  // STORE_EIGEN_MATRIX
            if (_solver.info() != Eigen::Success)
                fprintf(stderr,
                        "[ERROR] "
                        "EigenSolverCholeskyLLt::EigenSolverCholeskyLLt Failed "
                        "to factorize matrix\n"),
                        exit(0);
        }
        _eigenB.resize(M.rows());
    }
    void update(const SparseMatrixInterface<Real, MatrixRowIterator>& M) {
#ifdef STORE_EIGEN_MATRIX
#pragma omp parallel for
        for (int i = 0; i < M.rows(); i++)
            for (MatrixRowIterator iter = M.begin(i); iter != M.end(i); iter++)
                _eigenM.coeffRef(i, iter->N) = iter->Value;
        _solver.factorize(_eigenM);
#else   // !STORE_EIGEN_MATRIX
        Eigen::SparseMatrix<double> eigenM(int(M.rows()), int(M.rows()));
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(M.entries());
        for (int i = 0; i < M.rows(); i++)
            for (MatrixRowIterator iter = M.begin(i); iter != M.end(i); iter++)
                triplets.push_back(
                        Eigen::Triplet<double>(i, iter->N, iter->Value));
        eigenM.setFromTriplets(triplets.begin(), triplets.end());
        _solver.factorize(eigenM);
#endif  // STORE_EIGEN_MATRIX
        switch (_solver.info()) {
            case Eigen::Success:
                break;
            case Eigen::NumericalIssue:
                fprintf(stderr,
                        "[ERROR] EigenSolverCholeskyLLt::update Failed to "
                        "factorize matrix (numerical issue)\n"),
                        exit(0);
            case Eigen::NoConvergence:
                fprintf(stderr,
                        "[ERROR] EigenSolverCholeskyLLt::update Failed to "
                        "factorize matrix (no convergence)\n"),
                        exit(0);
            case Eigen::InvalidInput:
                fprintf(stderr,
                        "[ERROR] EigenSolverCholeskyLLt::update Failed to "
                        "factorize matrix (invalid input)\n"),
                        exit(0);
            default:
                fprintf(stderr,
                        "[ERROR] EigenSolverCholeskyLLt::update Failed to "
                        "factorize matrix\n"),
                        exit(0);
        }
    }
    void solve(ConstPointer(Real) b, Pointer(Real) x) {
#pragma omp parallel for
        for (int i = 0; i < _eigenB.size(); i++) _eigenB[i] = b[i];
        Eigen_Vector eigenX = _solver.solve(_eigenB);
#pragma omp parallel for
        for (int i = 0; i < eigenX.size(); i++) x[i] = (Real)eigenX[i];
    }
    size_t dimension(void) const { return _eigenB.size(); }
    static void Solve(const SparseMatrixInterface<Real, MatrixRowIterator>& M,
                      ConstPointer(Real) b,
                      Pointer(Real) x) {
        EigenSolverCholeskyLLt solver(M);
        solver.solve(b, x);
    }
};
template <class Real, class MatrixRowIterator>
class EigenSolverCholeskyLDLt : public EigenSolver<Real, MatrixRowIterator> {
#ifdef EIGEN_USE_MKL_ALL
    typedef Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> Eigen_Solver;
    typedef Eigen::VectorXd Eigen_Vector;
#else   // !EIGEN_USE_MKL_ALL
    typedef Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> Eigen_Solver;
    typedef Eigen::VectorXd Eigen_Vector;
#endif  // EIGEN_USE_MKL_ALL
    Eigen_Solver _solver;
    Eigen_Vector _eigenB;

public:
    EigenSolverCholeskyLDLt(
            const SparseMatrixInterface<Real, MatrixRowIterator>& M,
            bool analyzeOnly = false) {
        Eigen::SparseMatrix<double> eigenM(int(M.rows()), int(M.rows()));
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(M.entries());
        for (int i = 0; i < M.rows(); i++)
            for (MatrixRowIterator iter = M.begin(i); iter != M.end(i); iter++)
                triplets.push_back(
                        Eigen::Triplet<double>(i, iter->N, iter->Value));
        eigenM.setFromTriplets(triplets.begin(), triplets.end());
        _solver.analyzePattern(eigenM);
        if (!analyzeOnly) {
            _solver.factorize(eigenM);
            if (_solver.info() != Eigen::Success)
                fprintf(stderr,
                        "[ERROR] "
                        "EigenSolverCholeskyLDLt::EigenSolverCholeskyLDLt "
                        "Failed to factorize matrix\n"),
                        exit(0);
        }
        _eigenB.resize(M.rows());
    }
    void update(const SparseMatrixInterface<Real, MatrixRowIterator>& M) {
        Eigen::SparseMatrix<double> eigenM(int(M.rows()), int(M.rows()));
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(M.entries());
        for (int i = 0; i < M.rows(); i++)
            for (MatrixRowIterator iter = M.begin(i); iter != M.end(i); iter++)
                triplets.push_back(
                        Eigen::Triplet<double>(i, iter->N, iter->Value));
        eigenM.setFromTriplets(triplets.begin(), triplets.end());
        _solver.factorize(eigenM);
        if (_solver.info() != Eigen::Success)
            fprintf(stderr,
                    "[ERROR] EigenSolverCholeskyLDLt::update Failed to "
                    "factorize matrix\n"),
                    exit(0);
    }
    void solve(ConstPointer(Real) b, Pointer(Real) x) {
#pragma omp parallel for
        for (int i = 0; i < _eigenB.size(); i++) _eigenB[i] = b[i];
        Eigen_Vector eigenX = _solver.solve(_eigenB);
#pragma omp parallel for
        for (int i = 0; i < eigenX.size(); i++) x[i] = (Real)eigenX[i];
    }
    size_t dimension(void) const { return _eigenB.size(); }
    static void Solve(const SparseMatrixInterface<Real, MatrixRowIterator>& M,
                      ConstPointer(Real) b,
                      Pointer(Real) x) {
        EigenSolverCholeskyLDLt solver(M);
        solver.solve(b, x);
    }
};
template <class Real, class MatrixRowIterator>
class EigenSolverCG : public EigenSolver<Real, MatrixRowIterator> {
#if 1
    //	Eigen::ConjugateGradient< Eigen::SparseMatrix< double > , Eigen::Lower ,
    // Eigen::IncompleteLUT< double > > _solver;
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> _solver;
#else
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> _solver;
#endif
    Eigen::VectorXd _eigenB, _eigenX;
    Eigen::SparseMatrix<double> _eigenM;

public:
    EigenSolverCG(const SparseMatrixInterface<Real, MatrixRowIterator>& M,
                  int iters = 20,
                  double tolerance = 0.) {
        _eigenM.resize((int)M.rows(), (int)M.rows());
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(M.entries());
        for (int i = 0; i < M.rows(); i++)
            for (MatrixRowIterator iter = M.begin(i); iter != M.end(i); iter++)
                triplets.push_back(
                        Eigen::Triplet<double>(i, iter->N, iter->Value));
        _eigenM.setFromTriplets(triplets.begin(), triplets.end());
        _solver.compute(_eigenM);
        _solver.analyzePattern(_eigenM);
        if (_solver.info() != Eigen::Success)
            fprintf(stderr,
                    "[ERROR] EigenSolverCG::EigenSolverCG Failed to factorize "
                    "matrix\n"),
                    exit(0);
        _eigenB.resize(M.rows()), _eigenX.resize(M.rows());
        _solver.setMaxIterations(iters);
        _solver.setTolerance(tolerance);
    }
    void update(const SparseMatrixInterface<Real, MatrixRowIterator>& M) {
#pragma omp parallel for
        for (int i = 0; i < M.rows(); i++)
            for (MatrixRowIterator iter = M.begin(i); iter != M.end(i); iter++)
                _eigenM.coeffRef(i, iter->N) = iter->Value;
        _solver.compute(_eigenM);
        _solver.analyzePattern(_eigenM);
        if (_solver.info() != Eigen::Success)
            fprintf(stderr,
                    "[ERROR] EigenSolverCG::update Failed to factorize "
                    "matrix\n"),
                    exit(0);
    }

    void setIters(int iters) { _solver.setMaxIterations(iters); }
    void solve(ConstPointer(Real) b, Pointer(Real) x) {
#pragma omp parallel for
        for (int i = 0; i < _eigenB.size(); i++)
            _eigenB[i] = b[i], _eigenX[i] = x[i];
        _eigenX = _solver.solveWithGuess(_eigenB, _eigenX);
#pragma omp parallel for
        for (int i = 0; i < _eigenX.size(); i++) x[i] = _eigenX[i];
    }
    size_t dimension(void) const { return _eigenB.size(); }
    static void Solve(const SparseMatrixInterface<Real, MatrixRowIterator>& M,
                      const Real* b,
                      Real* x,
                      int iters) {
        EigenSolverCG solver(M, iters);
        solver.solve(b, x);
    }
};

#endif  // USE_EIGEN

#endif  // LINEAR_SOLVERS_INCLUDE