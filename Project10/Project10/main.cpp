#include <string>
#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videostab.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::videostab;

void processing(Ptr<IFrameSource> stabilizedFrames, string outputPath) {
	VideoWriter writer;
	Mat stabilizedFrame;
	int nframes = 0;
	double outputFps = 25;
	while (!(stabilizedFrame = stabilizedFrames->nextFrame()).empty()) {
		nframes++;
		if (!outputPath.empty()) {
			if (!writer.isOpened()) writer.open(outputPath, VideoWriter::fourcc('X', 'V', 'I', 'D'), outputFps, stabilizedFrame.size());
			writer << stabilizedFrame;
		}
		imshow("stabilizedFrame", stabilizedFrame);
		char key = static_cast<char>(waitKey(3));
		if (key == 27) { cout << endl; break; }
	}
	cout << "processed frames: " << nframes << endl;
	cout << "finished " << endl;
}

int main() {
	Ptr<IFrameSource> stabilizedFrames;
	try	{
		string inputPath;
		string outputPath;
		
		inputPath = "video.avi";
		outputPath = ".\\cube4_stabilized.avi";

		Ptr<VideoFileSource> source = makePtr<VideoFileSource>(inputPath);
		cout << "frame count (rough): " << source->count() << endl;

		double min_inlier_ratio = 0.1;
		Ptr<MotionEstimatorRansacL2> est = makePtr<MotionEstimatorRansacL2>(MM_AFFINE);
		RansacParams ransac = est->ransacParams();
		ransac.size = 3;
		ransac.thresh = 5;
		ransac.eps = 0.5;
		est->setRansacParams(ransac);
		est->setMinInlierRatio(min_inlier_ratio);

		int nkps = 1000;
		Ptr<GFTTDetector> feature_detector = GFTTDetector::create(nkps);

		Ptr<KeypointBasedMotionEstimator> motionEstBuilder = makePtr<KeypointBasedMotionEstimator>(est);
		motionEstBuilder->setDetector(feature_detector);
		Ptr<IOutlierRejector> outlierRejector = makePtr<NullOutlierRejector>();
		motionEstBuilder->setOutlierRejector(outlierRejector);
		StabilizerBase *stabilizer = 0;
		bool isTwoPass = 1;
		int radius_pass = 15;
		if (isTwoPass) {
			bool est_trim = true;
			TwoPassStabilizer *twoPassStabilizer = new TwoPassStabilizer();
			twoPassStabilizer->setEstimateTrimRatio(est_trim);
			twoPassStabilizer->setMotionStabilizer(makePtr<GaussianMotionFilter>(radius_pass));
			stabilizer = twoPassStabilizer;
		}
		else {
			OnePassStabilizer *onePassStabilizer = new OnePassStabilizer();
			onePassStabilizer->setMotionFilter(makePtr<GaussianMotionFilter>(radius_pass));
			stabilizer = onePassStabilizer;
		}
		int radius = 15;
		double trim_ratio = 0.1;
		bool incl_constr = false;
		stabilizer->setFrameSource(source);
		stabilizer->setMotionEstimator(motionEstBuilder);
		stabilizer->setRadius(radius);
		stabilizer->setTrimRatio(trim_ratio);
		stabilizer->setCorrectionForInclusion(incl_constr);
		stabilizer->setBorderMode(BORDER_REPLICATE);
		stabilizedFrames.reset(dynamic_cast<IFrameSource*>(stabilizer));
		processing(stabilizedFrames, outputPath);
	}
	catch (const exception &e) {
		
		cout << "error: " << e.what() << endl;
		stabilizedFrames.release();
		return -1;
	}
	stabilizedFrames.release();
	return 0;
}