#include <Eigen/Dense>
#include <map>
#include <vector>
#include <fstream>
#include <cfloat>

struct EigenVectorXdCompare
{
    bool operator()(const Eigen::VectorXd& a, const Eigen::VectorXd& b) const
    {
        assert(a.size()==b.size());
        for(Eigen::VectorXd::Index i=0;i<a.size();++i)
        {
            if(a[i]<b[i]) return true;
            if(a[i]>b[i]) return false;
        }
        return false;
    }
};

struct EigenVectorXiCompare
{
    bool operator()(const Eigen::VectorXi& a, const Eigen::VectorXi& b) const
    {
        assert(a.size()==b.size());
        for(Eigen::VectorXi::Index i=0;i<a.size();++i)
        {
            if(a[i]<b[i]) return true;
            if(a[i]>b[i]) return false;
        }
        return false;
    }
};

static std::string toString(const Eigen::MatrixXd& mat){
    std::stringstream ss;
    ss << mat;
    return ss.str();
}

template<typename Func, typename... Ts>
Eigen::VectorXd minimize_discrete_stepwise_cpp(Func fun, Eigen::VectorXd x, 
    std::map<Eigen::VectorXd,double,EigenVectorXdCompare>& results_cache, 
    Eigen::VectorXd dx, int search_range = 1, int maxiter = 10000, Ts&& ...args)
{
    /*Minimizes a discrete function by nearest neighbour descent.

    Parameters
    ----------
    fun : `callable`
        Function to be minimized.
        Its signature must be `fun(x0, *args) -> float`.
    x0 : `Array[1]` or `Scalar`
        Initial guess for solution.
    args : `tuple(Any)`
        Additional function arguments.
    kwargs : `dict(str->Any)`
        Function keywword arguments.
    dx : `Array[1]` or `Scalar`
        Discrete steps along each dimension.
        If scalar, applies given step to all dimensions.
    search_range : `int`
        Number of discrete steps to be evaluated per iteration.
        E.g. `search_range = 1` means evaluating in the range `[-1, 0, 1]`.
        Larger `search_range` avoids ending in local optimum but is slower.
    maxiter : `int`
        Maximum number of optimization steps.
    results_cache : `dict` or `None`
        Dictionary of pre-calculated results.
    ret_cache : `bool`
        Whether to return the `results_cache`.

    Returns
    -------
    x : `Array[1, float]` or `float`
        Solution. Scalar or vectorial depending on `x0`.
    results_cache : `dict`
        Results cache. Only returned if `ret_cache is True`.*/

    // Parse parameters
    if (x.size() == 1 && dx.size() > 0)
    {
        double val = x[0];
        x = Eigen::VectorXd(dx.size());
        for(int i = 0; i < dx.size(); i++)
        {
            x[i] = val;
        }
    }
    else if (dx.size() == 1 && x.size() > 0)
    {
        double val = dx[0];
        dx = Eigen::VectorXd(x.size());
        for(int i = 0; i < x.size(); i++)
        {
            dx[i] = val;
        }
    }
    // Initialize optimization variables
    size_t sizeMg = std::pow(2 * search_range + 1, dx.size());
    Eigen::MatrixXd dxArMg(sizeMg, dx.size());
    std::vector<int> ranges;
    for(int j = 0; j < dx.size(); j++)
    {
        ranges.push_back(std::pow(2 * search_range + 1, dx.size() - j - 1));
    }
    for(size_t i = 0; i < sizeMg; i++)
    {
        for(Eigen::Index j = 0; j < dx.size(); j++)
        {
            int searchRangeValue = i % (ranges[j] * (2 * search_range + 1)) / ranges[j] - search_range;
            dxArMg(i, j) = dx[j] * searchRangeValue;
        }
    }
    // Perform optimization
    bool converged = false;
    for(int i = 0; i < maxiter; i++)
    {
        // Get result for each step direction
        std::vector<double> resMg(sizeMg, NAN);
        Eigen::MatrixXd xMg = dxArMg.rowwise() + x.transpose();     // [nsteps, ndim]
        double resMin = DBL_MAX;
        int idxMin = -1;
        for(size_t idx = 0; idx < sizeMg; idx++)
        {
            Eigen::VectorXd key = xMg.row(idx);
            double result = 0;
            if(results_cache.count(key) == 0)
            {
                result = fun(key, args...);
                results_cache[key] = result;
            }
            else
            {
                result = results_cache[key];
            }
            if(result < resMin)
            {
                idxMin = (int)idx;
                resMin = result;
            }
        }
        x = xMg.row(idxMin);
        // Check convergence
        if(dxArMg.row(idxMin).isZero())
        {
            converged = true;
            break;
        }
    }
    // Return results
    if(!converged)
    {
        throw std::invalid_argument("maxiter reached without convergence (x = {" + toString(x) + "})");
    }
    return x;
}

template<typename Func, typename... Ts>
Eigen::VectorXd maximize_discrete_stepwise_cpp(Func fun, Eigen::VectorXd x, 
    std::map<Eigen::VectorXd,double,EigenVectorXdCompare>& results_cache, 
    Eigen::VectorXd dx, int search_range = 1, int maxiter = 10000, Ts&& ...args)
{
    return minimize_discrete_stepwise_cpp([&fun](Eigen::VectorXd a, Ts... args)
    {
        return -fun(a, args...);
    }, x, results_cache, dx, search_range, maxiter, args...);
}