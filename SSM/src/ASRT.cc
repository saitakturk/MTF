#include "mtf/SSM/ASRT.h"
#include "mtf/SSM/SSMEstimator.h"
#include "mtf/Utilities/warpUtils.h"
#include "mtf/Utilities/miscUtils.h"
#include "opencv2/core/core_c.h"
#include "opencv2/calib3d/calib3d.hpp"

#define validate_asrt_warp(warp) \
	assert(warp(0, 1) == -warp(1, 0)); \
	assert(warp(2, 0) == 0 && warp(2, 1) == 0); \
	assert(warp(2, 2) == 1)

#define ASRT_NORMALIZED_INIT false
#define ASRT_PT_BASED_SAMPLING 0
#define ASRT_DEBUG_MODE 0

_MTF_BEGIN_NAMESPACE

ASRTParams::ASRTParams(const SSMParams *ssm_params,
bool _normalized_init, int _pt_based_sampling,
bool _debug_mode) :
SSMParams(ssm_params),
normalized_init(_normalized_init),
pt_based_sampling(_pt_based_sampling),
debug_mode(_debug_mode){}

ASRTParams::ASRTParams(const ASRTParams *params) :
SSMParams(params),
normalized_init(ASRT_NORMALIZED_INIT),
pt_based_sampling(ASRT_PT_BASED_SAMPLING),
debug_mode(ASRT_DEBUG_MODE){
	if(params){
		normalized_init = params->normalized_init;
		pt_based_sampling = params->pt_based_sampling;
		debug_mode = params->debug_mode;
	}
}

ASRT::ASRT(const ParamType *_params) :
ProjectiveBase(_params),
params(_params){

	printf("\n");
	printf("Using Anisotropic Scaling, Rotation and Translation SSM with:\n");
	printf("resx: %d\n", resx);
	printf("resy: %d\n", resy);
	printf("normalized_init: %d\n", params.normalized_init);
	printf("pt_based_sampling: %d\n", params.pt_based_sampling);
	printf("debug_mode: %d\n", params.debug_mode);

	name = "asrt";
	state_size = 5;
	curr_state.resize(state_size);

	utils::getNormUnitSquarePts(norm_pts, norm_corners, resx, resy,
		1 - resx / 2.0, 1 - resy / 2.0, resx / 2.0, resy / 2.0);
	utils::homogenize(norm_pts, norm_pts_hm);
	utils::homogenize(norm_corners, norm_corners_hm);

	init_corners = getNormCorners();
	init_corners_hm = getHomNormCorners();
	init_pts = getNormPts();
	init_pts_hm = getHomNormPts();
}

void ASRT::setCorners(const CornersT& corners){
	if(params.normalized_init){
		curr_warp = utils::computeASRTDLT(init_corners, corners);
		getStateFromWarp(curr_state, curr_warp);

		curr_pts.noalias() = curr_warp.topRows<2>() * init_pts_hm;
		curr_corners.noalias() = curr_warp.topRows<2>() * init_corners_hm;

		utils::homogenize(curr_pts, curr_pts_hm);
		utils::homogenize(curr_corners, curr_corners_hm);
	} else {
		curr_corners = corners;
		utils::homogenize(curr_corners, curr_corners_hm);

		getPtsFromCorners(curr_warp, curr_pts, curr_pts_hm, curr_corners);

		init_corners = curr_corners;
		init_pts = curr_pts;
		utils::homogenize(init_corners, init_corners_hm);
		utils::homogenize(init_pts, init_pts_hm);

		curr_warp = Matrix3d::Identity();
		curr_state.fill(0);
	}
}

void ASRT::setState(const VectorXd &ssm_state){
	validate_ssm_state(ssm_state);
	curr_state = ssm_state;
	getWarpFromState(curr_warp, curr_state);
	curr_pts.noalias() = curr_warp.topRows<2>() * init_pts_hm;
	curr_corners.noalias() = curr_warp.topRows<2>() * init_corners_hm;
}

void ASRT::compositionalUpdate(const VectorXd& state_update){
	validate_ssm_state(state_update);

	getWarpFromState(warp_update_mat, state_update);
	curr_warp = curr_warp * warp_update_mat;

	getStateFromWarp(curr_state, curr_warp);

	curr_pts.noalias() = curr_warp.topRows<2>() * init_pts_hm;
	curr_corners.noalias() = curr_warp.topRows<2>() * init_corners_hm;
}

void ASRT::getWarpFromState(Matrix3d &warp_mat,
	const VectorXd& ssm_state){
	validate_ssm_state(ssm_state);

	double tx = ssm_state(0);
	double ty = ssm_state(1);
	double a = ssm_state(2);
	double b = ssm_state(3);
	double c = ssm_state(4);

	warp_mat(0, 0) = 1 + a;
	warp_mat(0, 1) = -b;
	warp_mat(0, 2) = tx;
	warp_mat(1, 0) = b;
	warp_mat(1, 1) = 1 + c;
	warp_mat(1, 2) = ty;
	warp_mat(2, 0) = 0;
	warp_mat(2, 1) = 0;
	warp_mat(2, 2) = 1;
}


void ASRT::getStateFromWarp(VectorXd &state_vec,
	const Matrix3d& sim_mat){
	validate_ssm_state(state_vec);
	validate_asrt_warp(sim_mat);

	state_vec(0) = sim_mat(0, 2);
	state_vec(1) = sim_mat(1, 2);
	state_vec(2) = sim_mat(0, 0) - 1;
	state_vec(3) = sim_mat(1, 0);
	state_vec(4) = sim_mat(1, 1) - 1;
}

void ASRT::getInitPixGrad(Matrix2Xd &dI_dp, int pt_id) {
	double x = init_pts(0, pt_id);
	double y = init_pts(1, pt_id);
	dI_dp <<
		1, 0, x, -y, 0,
		0, 1, 0, x, y;
}

void ASRT::cmptInitPixJacobian(MatrixXd &dI_dp,
	const PixGradT &dI_dx){
	validate_ssm_jacobian(dI_dp, dI_dx);

	unsigned int ch_pt_id = 0;
	for(unsigned int pt_id = 0; pt_id < n_pts; ++pt_id){
		spi_pt_check_mc(spi_mask, pt_id, ch_pt_id);

		double x = init_pts(0, pt_id);
		double y = init_pts(1, pt_id);
		for(unsigned int ch_id = 0; ch_id < n_channels; ++ch_id){
			double Ix = dI_dx(ch_pt_id, 0);
			double Iy = dI_dx(ch_pt_id, 1);

			dI_dp(ch_pt_id, 0) = Ix;
			dI_dp(ch_pt_id, 1) = Iy;
			dI_dp(ch_pt_id, 2) = Ix*x;
			dI_dp(ch_pt_id, 3) = Iy*x - Ix*y;
			dI_dp(ch_pt_id, 4) = Iy*y;
			++ch_pt_id;
		}
	}
}

void ASRT::cmptWarpedPixJacobian(MatrixXd &dI_dp,
	const PixGradT &dI_dx){
	validate_ssm_jacobian(dI_dp, dI_dx);
	double a = curr_state(2) + 1, b = -curr_state(3);
	double c = curr_state(3), d = curr_state(4) + 1;

	unsigned int ch_pt_id = 0;
	for(unsigned int pt_id = 0; pt_id < n_pts; ++pt_id){
		spi_pt_check_mc(spi_mask, pt_id, ch_pt_id);

		double x = init_pts(0, pt_id);
		double y = init_pts(1, pt_id);

		for(unsigned int ch_id = 0; ch_id < n_channels; ++ch_id){
			double Ix = a*dI_dx(ch_pt_id, 0) + c*dI_dx(ch_pt_id, 1);
			double Iy = b*dI_dx(ch_pt_id, 0) + d*dI_dx(ch_pt_id, 1);

			dI_dp(ch_pt_id, 0) = Ix;
			dI_dp(ch_pt_id, 1) = Iy;
			dI_dp(ch_pt_id, 2) = Ix*x;
			dI_dp(ch_pt_id, 3) = Iy*x - Ix*y;
			dI_dp(ch_pt_id, 4) = Iy*y;
			++ch_pt_id;
		}
	}
}

void ASRT::cmptApproxPixJacobian(MatrixXd &dI_dp, const PixGradT &dI_dx) {
	validate_ssm_jacobian(dI_dp, dI_dx);
	double a_plus_1 = curr_state(2) + 1, b = curr_state(3), c_plus_1 = curr_state(4) + 1;
	double inv_det = 1.0 / (a_plus_1*c_plus_1 + b*b);

	unsigned int ch_pt_id = 0;
	for(unsigned int pt_id = 0; pt_id < n_pts; ++pt_id){
		spi_pt_check_mc(spi_mask, pt_id, ch_pt_id);

		double x = init_pts(0, pt_id);
		double y = init_pts(1, pt_id);
		for(unsigned int ch_id = 0; ch_id < n_channels; ++ch_id){
			double Ix = (dI_dx(ch_pt_id, 0)*a_plus_1 - dI_dx(ch_pt_id, 1)*b) * inv_det;
			double Iy = (dI_dx(ch_pt_id, 0)*b + dI_dx(ch_pt_id, 1)*c_plus_1) * inv_det;

			dI_dp(ch_pt_id, 0) = Ix;
			dI_dp(ch_pt_id, 1) = Iy;
			dI_dp(ch_pt_id, 2) = Ix*x;
			dI_dp(ch_pt_id, 3) = Iy*x - Ix*y;
			dI_dp(ch_pt_id, 4) = Iy*y;

			++ch_pt_id;
		}
	}
}

void ASRT::cmptInitPixHessian(MatrixXd &d2I_dp2, const PixHessT &d2I_dw2,
	const PixGradT &dI_dw){
	validate_ssm_hessian(d2I_dp2, d2I_dw2, dI_dw);

	unsigned int ch_pt_id = 0;
	for(unsigned int pt_id = 0; pt_id < n_pts; pt_id++){
		spi_pt_check_mc(spi_mask, pt_id, ch_pt_id);

		double x = init_pts(0, pt_id);
		double y = init_pts(1, pt_id);
		Matrix25d dw_dp;
		dw_dp <<
			1, 0, x, -y, 0,
			0, 1, 0, x, y;
		for(unsigned int ch_id = 0; ch_id < n_channels; ++ch_id){
			Map<Matrix5d>(d2I_dp2.col(ch_pt_id).data()) = dw_dp.transpose()*
				Map<const Matrix2d>(d2I_dw2.col(ch_pt_id).data())*dw_dp;
			++ch_pt_id;
		}
	}
}
void ASRT::cmptWarpedPixHessian(MatrixXd &d2I_dp2, const PixHessT &d2I_dw2,
	const PixGradT &dI_dw) {
	validate_ssm_hessian(d2I_dp2, d2I_dw2, dI_dw);
	double a2 = curr_state(2) + 1, a3 = curr_state(3), a4 = curr_state(4) + 1;
	Matrix2d dw_dx;
	dw_dx <<
		a2, -a3,
		a3, a4;

	unsigned int ch_pt_id = 0;
	for(unsigned int pt_id = 0; pt_id < n_pts; ++pt_id) {
		spi_pt_check_mc(spi_mask, pt_id, ch_pt_id);

		double x = init_pts(0, pt_id);
		double y = init_pts(1, pt_id);

		Matrix25d dw_dp;
		dw_dp <<
			1, 0, x, -y, 0,
			0, 1, 0, x, y;

		for(unsigned int ch_id = 0; ch_id < n_channels; ++ch_id){
			Map<Matrix5d>(d2I_dp2.col(ch_pt_id).data()) = dw_dp.transpose()*
				dw_dx.transpose()*Map<const Matrix2d>(d2I_dw2.col(ch_pt_id).data())*dw_dx*dw_dp;
			++ch_pt_id;
		}
	}
}

void ASRT::estimateWarpFromCorners(VectorXd &state_update, const Matrix24d &in_corners,
	const Matrix24d &out_corners){
	validate_ssm_state(state_update);

	Matrix3d warp_update_mat = utils::computeASRTDLT(in_corners, out_corners);
	getStateFromWarp(state_update, warp_update_mat);
}

void ASRT::estimateWarpFromPts(VectorXd &state_update, vector<uchar> &mask,
	const vector<cv::Point2f> &in_pts, const vector<cv::Point2f> &out_pts,
	const EstimatorParams &est_params){
	cv::Mat asrt_params = estimateASRT(in_pts, out_pts, mask, est_params);
	state_update(0) = asrt_params.at<double>(3, 0);
	state_update(1) = asrt_params.at<double>(4, 0);
	state_update(2) = asrt_params.at<double>(0, 0) - 1;
	state_update(3) = asrt_params.at<double>(1, 0);
	state_update(4) = asrt_params.at<double>(2, 0) - 1;
}


void ASRT::updateGradPts(double grad_eps){
	Vector2d diff_vec_x_warped = curr_warp.topRows<2>().col(0) * grad_eps;
	Vector2d diff_vec_y_warped = curr_warp.topRows<2>().col(1) * grad_eps;

	for(unsigned int pt_id = 0; pt_id < n_pts; ++pt_id){
		spi_pt_check(spi_mask, pt_id);

		grad_pts(0, pt_id) = curr_pts(0, pt_id) + diff_vec_x_warped(0);
		grad_pts(1, pt_id) = curr_pts(1, pt_id) + diff_vec_x_warped(1);

		grad_pts(2, pt_id) = curr_pts(0, pt_id) - diff_vec_x_warped(0);
		grad_pts(3, pt_id) = curr_pts(1, pt_id) - diff_vec_x_warped(1);

		grad_pts(4, pt_id) = curr_pts(0, pt_id) + diff_vec_y_warped(0);
		grad_pts(5, pt_id) = curr_pts(1, pt_id) + diff_vec_y_warped(1);

		grad_pts(6, pt_id) = curr_pts(0, pt_id) - diff_vec_y_warped(0);
		grad_pts(7, pt_id) = curr_pts(1, pt_id) - diff_vec_y_warped(1);
	}
}


void ASRT::updateHessPts(double hess_eps){
	double hess_eps2 = 2 * hess_eps;

	Vector2d diff_vec_xx_warped = curr_warp.topRows<2>().col(0) * hess_eps2;
	Vector2d diff_vec_yy_warped = curr_warp.topRows<2>().col(1) * hess_eps2;
	Vector2d diff_vec_xy_warped = (curr_warp.topRows<2>().col(0) + curr_warp.topRows<2>().col(1)) * hess_eps;
	Vector2d diff_vec_yx_warped = (curr_warp.topRows<2>().col(0) - curr_warp.topRows<2>().col(1)) * hess_eps;

	for(unsigned int pt_id = 0; pt_id < n_pts; ++pt_id){

		spi_pt_check(spi_mask, pt_id);

		hess_pts(0, pt_id) = curr_pts(0, pt_id) + diff_vec_xx_warped(0);
		hess_pts(1, pt_id) = curr_pts(1, pt_id) + diff_vec_xx_warped(1);

		hess_pts(2, pt_id) = curr_pts(0, pt_id) - diff_vec_xx_warped(0);
		hess_pts(3, pt_id) = curr_pts(1, pt_id) - diff_vec_xx_warped(1);

		hess_pts(4, pt_id) = curr_pts(0, pt_id) + diff_vec_yy_warped(0);
		hess_pts(5, pt_id) = curr_pts(1, pt_id) + diff_vec_yy_warped(1);

		hess_pts(6, pt_id) = curr_pts(0, pt_id) - diff_vec_yy_warped(0);
		hess_pts(7, pt_id) = curr_pts(1, pt_id) - diff_vec_yy_warped(1);

		hess_pts(8, pt_id) = curr_pts(0, pt_id) + diff_vec_xy_warped(0);
		hess_pts(9, pt_id) = curr_pts(1, pt_id) + diff_vec_xy_warped(1);

		hess_pts(10, pt_id) = curr_pts(0, pt_id) - diff_vec_xy_warped(0);
		hess_pts(11, pt_id) = curr_pts(1, pt_id) - diff_vec_xy_warped(1);

		hess_pts(12, pt_id) = curr_pts(0, pt_id) + diff_vec_yx_warped(0);
		hess_pts(13, pt_id) = curr_pts(1, pt_id) + diff_vec_yx_warped(1);

		hess_pts(14, pt_id) = curr_pts(0, pt_id) - diff_vec_yx_warped(0);
		hess_pts(15, pt_id) = curr_pts(1, pt_id) - diff_vec_yx_warped(1);
	}
}
void ASRT::applyWarpToCorners(Matrix24d &warped_corners, const Matrix24d &orig_corners,
	const VectorXd &ssm_state){
	getWarpFromState(warp_mat, ssm_state);
	for(int corner_id = 0; corner_id < 4; corner_id++){
		warped_corners(0, corner_id) = warp_mat(0, 0)*orig_corners(0, corner_id) + warp_mat(0, 1)*orig_corners(1, corner_id) +
			warp_mat(0, 2);
		warped_corners(1, corner_id) = warp_mat(1, 0)*orig_corners(0, corner_id) + warp_mat(1, 1)*orig_corners(1, corner_id) +
			warp_mat(1, 2);
	}
}

void ASRT::applyWarpToPts(Matrix2Xd &warped_pts, const Matrix2Xd &orig_pts,
	const VectorXd &ssm_state){
	getWarpFromState(warp_mat, ssm_state);
	unsigned int n_pts = orig_pts.cols();
	for(unsigned int pt_id = 0; pt_id < n_pts; pt_id++){
		warped_pts(0, pt_id) = warp_mat(0, 0)*orig_pts(0, pt_id) + warp_mat(0, 1)*orig_pts(1, pt_id) +
			warp_mat(0, 2);
		warped_pts(1, pt_id) = warp_mat(1, 0)*orig_pts(0, pt_id) + warp_mat(1, 1)*orig_pts(1, pt_id) +
			warp_mat(1, 2);
	}
}

void ASRT::generatePerturbation(VectorXd &perturbation){
	assert(perturbation.size() == state_size);
	if(params.pt_based_sampling){
		PtsT orig_pts, perturbed_pts;
		orig_pts.resize(Eigen::NoChange, 3);
		perturbed_pts.resize(Eigen::NoChange, 3);
		//! use the bottom left, bottom right and top center points
		//! as canaonical points to add the random perturbations to;
		orig_pts.col(0) = init_corners.col(2);
		orig_pts.col(1) = init_corners.col(3);
		orig_pts.col(2) = (init_corners.col(0) + init_corners.col(1)) / 2.0;
		if(params.pt_based_sampling == 1){
			perturbed_pts(0, 0) = orig_pts(0, 0) + rand_dist[0](rand_gen[0]);
			perturbed_pts(1, 0) = orig_pts(1, 0) + rand_dist[1](rand_gen[1]);
			perturbed_pts(0, 1) = orig_pts(0, 1) + rand_dist[2](rand_gen[2]);
			perturbed_pts(1, 1) = orig_pts(1, 1) + rand_dist[3](rand_gen[3]);
			perturbed_pts(0, 2) = orig_pts(0, 2) + rand_dist[4](rand_gen[4]);
			perturbed_pts(1, 2) = orig_pts(1, 2) + rand_dist[4](rand_gen[4]);
		} else {
			//! different perturbation for x,y coordinates of each corner
			//! followed by consistent translational perturbation to all corners
			perturbed_pts(0, 0) = orig_pts(0, 0) + rand_dist[1](rand_gen[1]);
			perturbed_pts(1, 0) = orig_pts(1, 0) + rand_dist[1](rand_gen[1]);
			perturbed_pts(0, 1) = orig_pts(0, 1) + rand_dist[1](rand_gen[1]);
			perturbed_pts(1, 1) = orig_pts(1, 1) + rand_dist[1](rand_gen[1]);
			perturbed_pts(0, 2) = orig_pts(0, 2) + rand_dist[1](rand_gen[1]);
			perturbed_pts(1, 2) = orig_pts(1, 2) + rand_dist[1](rand_gen[1]);
			perturbed_pts = perturbed_pts.colwise() + Vector2d(rand_dist[0](rand_gen[0]), rand_dist[0](rand_gen[0]));
		}
		Matrix3d asrt_warp = utils::computeASRTDLT(orig_pts, perturbed_pts);
		getStateFromWarp(perturbation, asrt_warp);
	} else{
		ProjectiveBase::generatePerturbation(perturbation);
	}
}

cv::Mat ASRT::estimateASRT(cv::InputArray _in_pts, cv::InputArray _out_pts,
	cv::OutputArray _mask, const SSMEstimatorParams &params){
	cv::Mat in_pts = _in_pts.getMat(), out_pts = _out_pts.getMat();
	int n_pts = in_pts.checkVector(2);
	CV_Assert(n_pts >= 0 && out_pts.checkVector(2) == n_pts &&
		in_pts.type() == out_pts.type());

	cv::Mat H(5, 1, CV_64F);
	cv::Mat _pt1 = in_pts, _pt2 = out_pts;
	cv::Mat matH = H, c_mask, *p_mask = 0;
	if(_mask.needed()){
		_mask.create(n_pts, 1, CV_8U, -1, true);
		p_mask = &(c_mask = _mask.getMat());
	}
	bool ok = estimateASRT(&_pt1, &_pt2, &matH, p_mask, params) > 0;
	if(!ok)
		H = cv::Scalar(0);
	return H;
}

int	ASRT::estimateASRT(const cv::Mat* in_pts, const cv::Mat* out_pts,
	cv::Mat* __H, cv::Mat* mask, const SSMEstimatorParams &params) {
	bool result = false;
	cv::Ptr<cv::Mat> out_pts_hm, in_pts_hm, tempMask;

	double H[5];
	cv::Mat matH = cv::Mat(5, 1, CV_64FC1, H);

	CV_Assert(CV_IS_MAT(out_pts) && CV_IS_MAT(in_pts));

	int n_pts = MAX(out_pts->cols, out_pts->rows);
	CV_Assert(n_pts >= params.n_model_pts);

	out_pts_hm = cvCreateMat(1, n_pts, CV_64FC2);
	cvConvertPointsHomogeneous(out_pts, out_pts_hm);

	in_pts_hm = cvCreateMat(1, n_pts, CV_64FC2);
	cvConvertPointsHomogeneous(in_pts, in_pts_hm);

	if(mask) {
		CV_Assert(CV_IS_MASK_ARR(mask) && CV_IS_MAT_CONT(mask->type) &&
			(mask->rows == 1 || mask->cols == 1) &&
			mask->rows * mask->cols == n_pts);
	}
	if(mask || n_pts > params.n_model_pts)
		tempMask = cvCreateMat(1, n_pts, CV_8U);
	if(!tempMask.empty())
		cvSet(tempMask, cvScalarAll(1.));

	ASRTEstimator estimator(params.n_model_pts, params.use_boost_rng);

	int method = n_pts == params.n_model_pts ? 0 : params.method_cv;

	if(method == CV_LMEDS)
		result = estimator.runLMeDS(in_pts_hm, out_pts_hm, &matH, tempMask, params.confidence,
		params.max_iters, params.max_subset_attempts);
	else if(method == CV_RANSAC)
		result = estimator.runRANSAC(in_pts_hm, out_pts_hm, &matH, tempMask, params.ransac_reproj_thresh,
		params.confidence, params.max_iters, params.max_subset_attempts);
	else
		result = estimator.runKernel(in_pts_hm, out_pts_hm, &matH) > 0;

	if(result && n_pts > params.n_model_pts) {
		utils::icvCompressPoints((CvPoint2D64f*)in_pts_hm->data.ptr, tempMask->data.ptr, 1, n_pts);
		n_pts = utils::icvCompressPoints((CvPoint2D64f*)out_pts_hm->data.ptr, tempMask->data.ptr, 1, n_pts);
		in_pts_hm->cols = out_pts_hm->cols = n_pts;
		if(method == CV_RANSAC)
			estimator.runKernel(in_pts_hm, out_pts_hm, &matH);
		if(params.refine){
			estimator.refine(in_pts_hm, out_pts_hm, &matH, params.lm_max_iters);
		}
	}

	if(result)
		cvConvert(&matH, __H);

	if(mask && tempMask) {
		if(CV_ARE_SIZES_EQ(mask, tempMask))
			cvCopy(tempMask, mask);
		else
			cvTranspose(tempMask, mask);
	}

	return (int)result;
}



ASRTEstimator::ASRTEstimator(int _modelPoints, bool _use_boost_rng)
	: SSMEstimator(_modelPoints, cvSize(5, 1), 1, _use_boost_rng) {
	assert(_modelPoints >= 3);
	checkPartialSubsets = false;
}

int ASRTEstimator::runKernel(const cv::Mat* m1, const cv::Mat* m2, cv::Mat* H) {
	int n_pts = m1->rows * m1->cols;
	//if(n_pts != 3) {
	//    throw invalid_argument(cv::format("Invalid no. of points: %d provided", n_pts));
	//}
	const CvPoint2D64f* M = (const CvPoint2D64f*)m1->data.ptr;
	const CvPoint2D64f* m = (const CvPoint2D64f*)m2->data.ptr;

	Matrix2Xd in_pts, out_pts;
	in_pts.resize(Eigen::NoChange, n_pts);
	out_pts.resize(Eigen::NoChange, n_pts);
	for(int pt_id = 0; pt_id < n_pts; pt_id++) {
		in_pts(0, pt_id) = M[pt_id].x;
		in_pts(1, pt_id) = M[pt_id].y;

		out_pts(0, pt_id) = m[pt_id].x;
		out_pts(1, pt_id) = m[pt_id].y;
	}
	Matrix3d asrt_mat = utils::computeASRTDLT(in_pts, out_pts);

	double *H_ptr = H->data.db;
	H_ptr[0] = asrt_mat(0, 0);
	H_ptr[1] = asrt_mat(1, 0);
	H_ptr[2] = asrt_mat(1, 1);
	H_ptr[3] = asrt_mat(0, 2);
	H_ptr[4] = asrt_mat(1, 2);
	return 1;
}


void ASRTEstimator::computeReprojError(const cv::Mat* m1, const cv::Mat* m2,
	const cv::Mat* model, cv::Mat* _err) {
	int n_pts = m1->rows * m1->cols;
	const CvPoint2D64f* M = (const CvPoint2D64f*)m1->data.ptr;
	const CvPoint2D64f* m = (const CvPoint2D64f*)m2->data.ptr;
	const double* H = model->data.db;
	float* err = _err->data.fl;

	for(int pt_id = 0; pt_id < n_pts; pt_id++) {
		double dx = (H[0] * M[pt_id].x - H[1] * M[pt_id].y + H[3]) - m[pt_id].x;
		double dy = (H[1] * M[pt_id].x + H[2] * M[pt_id].y + H[4]) - m[pt_id].y;
		err[pt_id] = (float)(dx * dx + dy * dy);
	}
}

bool ASRTEstimator::refine(const cv::Mat* m1, const cv::Mat* m2,
	cv::Mat* model, int maxIters) {
	LevMarq solver(5, 0, cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, maxIters, DBL_EPSILON));
	int n_pts = m1->rows * m1->cols;
	const CvPoint2D64f* M = (const CvPoint2D64f*)m1->data.ptr;
	const CvPoint2D64f* m = (const CvPoint2D64f*)m2->data.ptr;
	cv::Mat modelPart = cv::Mat(solver.param->rows, solver.param->cols, model->type, model->data.ptr);
	cvCopy(&modelPart, solver.param);

	for(;;)	{
		const cv::Mat* _param = 0;
		cv::Mat *_JtJ = 0, *_JtErr = 0;
		double* _errNorm = 0;

		if(!solver.updateAlt(_param, _JtJ, _JtErr, _errNorm))
			break;

		for(int pt_id = 0; pt_id < n_pts; pt_id++)	{
			const double* h = _param->data.db;
			double Mx = M[pt_id].x, My = M[pt_id].y;
			double _xi = (h[0] * Mx - h[1] * My + h[3]);
			double _yi = (h[1] * Mx + h[2] * My + h[4]);
			double err[] = { _xi - m[pt_id].x, _yi - m[pt_id].y };
			if(_JtJ || _JtErr) {
				double J[][5] = {
					{ Mx, -My, 0, 1, 0 },
					{ 0, Mx, My, 0, 1 }
				};
				for(int j = 0; j < 5; j++) {
					for(int k = j; k < 5; k++)
						_JtJ->data.db[j * 5 + k] += J[0][j] * J[0][k] + J[1][j] * J[1][k];
					_JtErr->data.db[j] += J[0][j] * err[0] + J[1][j] * err[1];
				}
			}
			if(_errNorm)
				*_errNorm += err[0] * err[0] + err[1] * err[1];
		}
	}

	cvCopy(solver.param, &modelPart);
	return true;
}


_MTF_END_NAMESPACE

