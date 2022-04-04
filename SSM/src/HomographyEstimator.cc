#include "mtf/SSM//HomographyEstimator.h"
#include "mtf/Utilities/miscUtils.h"
#include "opencv2/core/core_c.h"
#include "opencv2/calib3d/calib3d.hpp"
#include <boost/random/uniform_int_distribution.hpp>


_MTF_BEGIN_NAMESPACE

HomographyEstimator::HomographyEstimator(int _modelPoints, bool _use_boost_rng)
	: SSMEstimator(_modelPoints, cvSize(3, 3), 1, _use_boost_rng) {
	assert(_modelPoints >= 4);
	checkPartialSubsets = false;
}

int HomographyEstimator::runKernel(const cv::Mat* m1, const cv::Mat* m2, cv::Mat* H) {
	int i, count = m1->rows * m1->cols;
	const CvPoint2D64f* M = (const CvPoint2D64f*)m1->data.ptr;
	const CvPoint2D64f* m = (const CvPoint2D64f*)m2->data.ptr;

	double LtL[9][9], W[9][1], V[9][9];
	cv::Mat _LtL = cv::Mat(9, 9, CV_64F, LtL);
	cv::Mat matW = cv::Mat(9, 1, CV_64F, W);
	cv::Mat matV = cv::Mat(9, 9, CV_64F, V);
	cv::Mat _H0 = cv::Mat(3, 3, CV_64F, V[8]);
	cv::Mat _Htemp = cv::Mat(3, 3, CV_64F, V[7]);
	CvPoint2D64f cM = { 0, 0 }, cm = { 0, 0 }, sM = { 0, 0 }, sm = { 0, 0 };

	for(i = 0; i < count; i++) {
		cm.x += m[i].x;
		cm.y += m[i].y;
		cM.x += M[i].x;
		cM.y += M[i].y;
	}

	cm.x /= count;
	cm.y /= count;
	cM.x /= count;
	cM.y /= count;

	for(i = 0; i < count; i++) {
		sm.x += fabs(m[i].x - cm.x);
		sm.y += fabs(m[i].y - cm.y);
		sM.x += fabs(M[i].x - cM.x);
		sM.y += fabs(M[i].y - cM.y);
	}

	if(fabs(sm.x) < DBL_EPSILON || fabs(sm.y) < DBL_EPSILON ||
		fabs(sM.x) < DBL_EPSILON || fabs(sM.y) < DBL_EPSILON)
		return 0;
	sm.x = count / sm.x;
	sm.y = count / sm.y;
	sM.x = count / sM.x;
	sM.y = count / sM.y;

	double invHnorm[9] = { 1. / sm.x, 0, cm.x, 0, 1. / sm.y, cm.y, 0, 0, 1 };
	double Hnorm2[9] = { sM.x, 0, -cM.x * sM.x, 0, sM.y, -cM.y * sM.y, 0, 0, 1 };
	cv::Mat _invHnorm = cv::Mat(3, 3, CV_64FC1, invHnorm);
	cv::Mat _Hnorm2 = cv::Mat(3, 3, CV_64FC1, Hnorm2);

	cvZero(&_LtL);
	for(i = 0; i < count; i++) {
		double x = (m[i].x - cm.x) * sm.x, y = (m[i].y - cm.y) * sm.y;
		double X = (M[i].x - cM.x) * sM.x, Y = (M[i].y - cM.y) * sM.y;
		double Lx[] = { X, Y, 1, 0, 0, 0, -x * X, -x * Y, -x };
		double Ly[] = { 0, 0, 0, X, Y, 1, -y * X, -y * Y, -y };
		int j, k;
		for(j = 0; j < 9; j++)
			for(k = j; k < 9; k++)
				LtL[j][k] += Lx[j] * Lx[k] + Ly[j] * Ly[k];
	}
	cvCompleteSymm(&_LtL);

	//cvSVD( &_LtL, &matW, 0, &matV, CV_SVD_MODIFY_A + CV_SVD_V_T );
	cvEigenVV(&_LtL, &matV, &matW);
	cv::MatMul(&_invHnorm, &_H0, &_Htemp);
	cv::MatMul(&_Htemp, &_Hnorm2, &_H0);
	cvConvertScale(&_H0, H, 1. / _H0.data.db[8]);

	return 1;
}


void HomographyEstimator::computeReprojError(const cv::Mat* m1, const cv::Mat* m2,
	const cv::Mat* model, cv::Mat* _err) {
	int i, count = m1->rows * m1->cols;
	const CvPoint2D64f* M = (const CvPoint2D64f*)m1->data.ptr;
	const CvPoint2D64f* m = (const CvPoint2D64f*)m2->data.ptr;
	const double* H = model->data.db;
	float* err = _err->data.fl;

	for(i = 0; i < count; i++) {
		double ww = 1. / (H[6] * M[i].x + H[7] * M[i].y + 1.);
		double dx = (H[0] * M[i].x + H[1] * M[i].y + H[2]) * ww - m[i].x;
		double dy = (H[3] * M[i].x + H[4] * M[i].y + H[5]) * ww - m[i].y;
		err[i] = (float)(dx * dx + dy * dy);
	}
}

bool HomographyEstimator::refine(const cv::Mat* m1, const cv::Mat* m2,
	cv::Mat* model, int maxIters) {
	LevMarq solver(8, 0, cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, maxIters, DBL_EPSILON));
	int i, j, k, count = m1->rows * m1->cols;
	const CvPoint2D64f* M = (const CvPoint2D64f*)m1->data.ptr;
	const CvPoint2D64f* m = (const CvPoint2D64f*)m2->data.ptr;
	cv::Mat modelPart = cv::Mat(solver.param->rows, solver.param->cols, model->type, model->data.ptr);
	cvCopy(&modelPart, solver.param);

	for(;;) {
		const cv::Mat* _param = 0;
		cv::Mat *_JtJ = 0, *_JtErr = 0;
		double* _errNorm = 0;

		if(!solver.updateAlt(_param, _JtJ, _JtErr, _errNorm))
			break;

		for(i = 0; i < count; i++) {
			const double* h = _param->data.db;
			double Mx = M[i].x, My = M[i].y;
			double ww = h[6] * Mx + h[7] * My + 1.;
			ww = fabs(ww) > DBL_EPSILON ? 1. / ww : 0;
			double _xi = (h[0] * Mx + h[1] * My + h[2]) * ww;
			double _yi = (h[3] * Mx + h[4] * My + h[5]) * ww;
			double err[] = { _xi - m[i].x, _yi - m[i].y };
			if(_JtJ || _JtErr) {
				double J[][8] = {
					{ Mx * ww, My * ww, ww, 0, 0, 0, -Mx*ww * _xi, -My*ww * _xi },
					{ 0, 0, 0, Mx * ww, My * ww, ww, -Mx*ww * _yi, -My*ww * _yi }
				};

				for(j = 0; j < 8; j++) {
					for(k = j; k < 8; k++)
						_JtJ->data.db[j * 8 + k] += J[0][j] * J[0][k] + J[1][j] * J[1][k];
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

cv::Mat estimateHomography(cv::InputArray _points1, cv::InputArray _points2,
	cv::OutputArray _mask, const SSMEstimatorParams &params){
	cv::Mat points1 = _points1.getMat(), points2 = _points2.getMat();
	int n_pts = points1.checkVector(2);
	CV_Assert(n_pts >= 0 && points2.checkVector(2) == n_pts &&
		points1.type() == points2.type());

	cv::Mat H(3, 3, CV_64F);
	cv::Mat _pt1 = points1, _pt2 = points2;
	cv::Mat matH = H, c_mask, *p_mask = 0;
	if(_mask.needed()){
		_mask.create(n_pts, 1, CV_8U, -1, true);
		p_mask = &(c_mask = _mask.getMat());
	}
	bool ok = estimateHomography(&_pt1, &_pt2, &matH, p_mask, params) > 0;
	if(!ok)
		H = cv::Scalar(0);
	return H;
}

int estimateHomography(const cv::Mat* objectPoints, const cv::Mat* imagePoints,
	cv::Mat* __H, cv::Mat* mask, const SSMEstimatorParams &params){
	bool result = false;
	cv::Ptr<cv::Mat> m, M, tempMask;

	double H[9];
	cv::Mat matH = cv::Mat(3, 3, CV_64FC1, H);

	CV_Assert(CV_IS_MAT(imagePoints) && CV_IS_MAT(objectPoints));

	int n_pts = MAX(imagePoints->cols, imagePoints->rows);
	CV_Assert(n_pts >= params.n_model_pts);

	m = cvCreateMat(1, n_pts, CV_64FC2);
	cvConvertPointsHomogeneous(imagePoints, m);

	M = cvCreateMat(1, n_pts, CV_64FC2);
	cvConvertPointsHomogeneous(objectPoints, M);

	if(mask){
		CV_Assert(CV_IS_MASK_ARR(mask) && CV_IS_MAT_CONT(mask->type) &&
			(mask->rows == 1 || mask->cols == 1) &&
			mask->rows*mask->cols == n_pts);
	}
	if(mask || n_pts > params.n_model_pts)
		tempMask = cvCreateMat(1, n_pts, CV_8U);
	if(!tempMask.empty())
		cvSet(tempMask, cvScalarAll(1.));

	HomographyEstimator estimator(params.n_model_pts, params.use_boost_rng);
	int method = n_pts == params.n_model_pts ? 0 : params.method_cv;
	if(method == CV_LMEDS)
		result = estimator.runLMeDS(M, m, &matH, tempMask, params.confidence, 
		params.max_iters, params.max_subset_attempts);
	else if(method == CV_RANSAC)
		result = estimator.runRANSAC(M, m, &matH, tempMask, params.ransac_reproj_thresh,
		params.confidence, params.max_iters, params.max_subset_attempts);
	else
		result = estimator.runKernel(M, m, &matH) > 0;

	if(result && n_pts > params.n_model_pts){
		utils::icvCompressPoints((CvPoint2D64f*)M->data.ptr, tempMask->data.ptr, 1, n_pts);
		n_pts = utils::icvCompressPoints((CvPoint2D64f*)m->data.ptr, tempMask->data.ptr, 1, n_pts);
		M->cols = m->cols = n_pts;
		if(method == CV_RANSAC)
			estimator.runKernel(M, m, &matH);
		if(params.refine){
			estimator.refine(M, m, &matH, params.lm_max_iters);
		}
	}

	if(result)
		cvConvert(&matH, __H);

	if(mask && tempMask){
		if(CV_ARE_SIZES_EQ(mask, tempMask))
			cvCopy(tempMask, mask);
		else
			cvTranspose(tempMask, mask);
	}

	return (int)result;
}



//int	cvFindHomography(const PtsT &objectPoints, const PtsT &imagePoints,
//	ProjWarpT & __H, EstimatorMethod method, double ransacReprojThreshold,
//	VectorXb &mask){
//	int count = objectPoints.cols();
//	assert(count >= 4);
//	assert(mask.size() == count);

//	const double confidence = 0.995;
//	const int maxIters = 2000;
//	const double defaultRANSACReprojThreshold = 3;
//	bool result = false;

//	HomPtsT m, M;
//	ProjWarpT matH;
//	if(ransacReprojThreshold <= 0)
//		ransacReprojThreshold = defaultRANSACReprojThreshold;

//	homogenize(imagePoints, m);
//	homogenize(objectPoints, M);
//
//	mask.fill(1);

//	AffineEstimator estimator(4);
//	if(count == 4)
//		method = EstimatorMethod::LSTSQR;

//	if(method == EstimatorMethod::LMEDS)
//		result = estimator.runLMeDS(M, m, matH, mask, confidence, maxIters);
//	else if(method == EstimatorMethod::RANSAC)
//		result = estimator.runRANSAC(M, m, matH, mask, ransacReprojThreshold, confidence, maxIters);
//	else
//		result = estimator.runKernel(M, m, matH) > 0;

//	if(result && count > 4){
//		compressPoints(m, mask, 1, count);
//		count = compressPoints(M, mask, 1, count);
//		if(method == EstimatorMethod::RANSAC)
//			estimator.runKernel(M, m, matH);
//		estimator.refine(M, m, matH, 10);
//	}

//	if(result)
//		__H = matH;

//	return result;
//}


_MTF_END_NAMESPACE
