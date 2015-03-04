

#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

#include "opencv2/highgui/highgui_c.h"
#include <stdio.h>
#include "opencv2/calib3d/calib3d_c.h"
#include "opencv2/calib3d.hpp"

#include <sys/time.h>
#include <sys/resource.h>
#include <stdlib.h>



using namespace std;
using namespace cv;
using namespace cv::detail;

static void printUsage()
{
    cout <<
        "Rotation model images stitcher.\n\n"
        "stitching_detailed video_file [flags]\n\n"
        "Flags:\n"
        "  --skip_frames\n"
        "      Down-sample video by a factor of skip_frames value. E.g. a value\n"
        "      of 2 given will select every other frame in the video\n"
        "  --mem_limit\n"
        "      Limit the application memory usage to some number of MB's.  E.g. \n"
        "      --mem_limit 2048 will limit the application to 2 GB of memory\n"
        "  --skip_to_segment\n"
        "     Skip to segment n for the stitching.  If you've run the program \n"
        "     once already and have seen the segments found and want to adjust\n"
        "     the setting for a particular segment, use --skip_to_segment.  E.g.\n"
        "     --skip_to_segment 2 will only process the 2nd segment.\n"
        "     --skip_to_segment 1 will only process the 1st segment.\n"
        "     Default is to process all segments.\n"
        "  --preview\n"
        "      Run stitching in the preview mode. Works faster than usual mode,\n"
        "      but output image will have lower resolution.\n"
        "  --try_cuda (yes|no)\n"
        "      Try to use CUDA. The default value is 'no'. All default values\n"
        "      are for CPU mode.\n"
        "  --try_ocl (yes|no)\n"
        "      Try to use OpenCL. The default value is 'no'. All default values\n"
        "      are for CPU mode.\n"
        "\nMotion Estimation Flags:\n"
        "  --work_megapix <float>\n"
        "      Resolution for image registration step. The default is 0.6 Mpx.\n"
        "  --features (surf|orb)\n"
        "      Type of features used for images matching. The default is surf.\n"
        "  --match_conf <float>\n"
        "      Confidence for feature matching step. The default is 0.65 for surf and 0.3 for orb.\n"
        "  --conf_thresh <float>\n"
        "      Threshold for two images are from the same panorama confidence.\n"
        "      The default is 1.0.\n"
        "  --ba (reproj|ray)\n"
        "      Bundle adjustment cost function. The default is ray.\n"
        "  --ba_refine_mask (mask)\n"
        "      Set refinement mask for bundle adjustment. It looks like 'x_xxx',\n"
        "      where 'x' means refine respective parameter and '_' means don't\n"
        "      refine one, and has the following format:\n"
        "      <fx><skew><ppx><aspect><ppy>. The default mask is 'xxxxx'. If bundle\n"
        "      adjustment doesn't support estimation of selected parameter then\n"
        "      the respective flag is ignored.\n"
        "  --wave_correct (no|horiz|vert)\n"
        "      Perform wave effect correction. The default is 'horiz'.\n"
        "  --save_graph <file_name>\n"
        "      Save matches graph represented in DOT language to <file_name> file.\n"
        "      Labels description: Nm is number of matches, Ni is number of inliers,\n"
        "      C is confidence.\n"
        "\nCompositing Flags:\n"
        "  --warp (plane|cylindrical|spherical|fisheye|stereographic|compressedPlaneA2B1|compressedPlaneA1.5B1|compressedPlanePortraitA2B1|compressedPlanePortraitA1.5B1|paniniA2B1|paniniA1.5B1|paniniPortraitA2B1|paniniPortraitA1.5B1|mercator|transverseMercator)\n"
        "      Warp surface type. The default is 'spherical'.\n"
        "  --seam_megapix <float>\n"
        "      Resolution for seam estimation step. The default is 0.1 Mpx.\n"
        "  --seam (no|voronoi|gc_color|gc_colorgrad)\n"
        "      Seam estimation method. The default is 'gc_color'.\n"
        "  --compose_megapix <float>\n"
        "      Resolution for compositing step. Use -1 for original resolution.\n"
        "      The default is -1.\n"
        "  --expos_comp (no|gain|gain_blocks)\n"
        "      Exposure compensation method. The default is 'gain_blocks'.\n"
        "  --blend (no|feather|multiband)\n"
        "      Blending method. The default is 'multiband'.\n"
        "  --blend_strength <float>\n"
        "      Blending strength from [0,100] range. The default is 5.\n"
        "  --output <result_img>\n"
        "      The default is 'result.jpg'.\n";
}


// Default command line args
vector<String> img_names;
bool preview = false;
bool try_cuda = false;
bool try_ocl = false;
double work_megapix = 0.3;  //seems to affect warping time/mem usage
double seam_megapix = 0.01;
double compose_megapix = -1;
float conf_thresh = 0.75;
string features_type = "surf";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string warp_type = "stereographic";
int expos_comp_type = ExposureCompensator::NO;
float match_conf = 0.9;
string seam_find_type = "gc_color";
int blend_type = Blender::MULTI_BAND;
float blend_strength = 5;
string result_name = "result";
//.jpg will be implied

string video_name;
vector<Mat> original_frames;
int result_name_counter = 1;
int frame_skip = 1;
long MAXMEM_MB = 3788; // 3.7 GB
int skip_to_segment = 0;  //default is to process all segments
int min_segment_size = 5;  //a segment must have > min_segment_size number of frames to be stitched
bool force = false;
bool all_frames = false;
bool skip_ba = false;
bool reduce_frames = true;
bool just_show_segments = false;

static int parseCmdArgs(int argc, char** argv)
{
    if (argc == 1)
    {
        printUsage();
        return -1;
    }
    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
        {
            printUsage();
            return -1;
        }
        else if (string(argv[i]) == "--preview")
        {
            preview = true;
        }
        else if (string(argv[i]) == "--try_cuda")
        {
            if (string(argv[i + 1]) == "no")
                try_cuda = false;
            else if (string(argv[i + 1]) == "yes")
                try_cuda = true;
            else
            {
                cout << "Bad --try_cuda flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--try_ocl")
        {
            if (string(argv[i + 1]) == "no")
                try_ocl = false;
            else if (string(argv[i + 1]) == "yes")
                try_ocl = true;
            else
            {
                cout << "Bad --try_ocl flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--work_megapix")
        {
            work_megapix = atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--seam_megapix")
        {
            seam_megapix = atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--compose_megapix")
        {
            compose_megapix = atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--result")
        {
            result_name = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--features")
        {
            features_type = argv[i + 1];
            if (features_type == "orb")
                match_conf = 0.3f;
            i++;
        }
        else if (string(argv[i]) == "--match_conf")
        {
            match_conf = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--conf_thresh")
        {
            conf_thresh = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--ba")
        {
            ba_cost_func = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--ba_refine_mask")
        {
            ba_refine_mask = argv[i + 1];
            if (ba_refine_mask.size() != 5)
            {
                cout << "Incorrect refinement mask length.\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--wave_correct")
        {
            if (string(argv[i + 1]) == "no")
                do_wave_correct = false;
            else if (string(argv[i + 1]) == "horiz")
            {
                do_wave_correct = true;
                wave_correct = detail::WAVE_CORRECT_HORIZ;
            }
            else if (string(argv[i + 1]) == "vert")
            {
                do_wave_correct = true;
                wave_correct = detail::WAVE_CORRECT_VERT;
            }
            else
            {
                cout << "Bad --wave_correct flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--save_graph")
        {
            save_graph = true;
            save_graph_to = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--warp")
        {
            warp_type = string(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--expos_comp")
        {
            if (string(argv[i + 1]) == "no")
                expos_comp_type = ExposureCompensator::NO;
            else if (string(argv[i + 1]) == "gain")
                expos_comp_type = ExposureCompensator::GAIN;
            else if (string(argv[i + 1]) == "gain_blocks")
                expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
            else
            {
                cout << "Bad exposure compensation method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--seam")
        {
            if (string(argv[i + 1]) == "no" ||
                string(argv[i + 1]) == "voronoi" ||
                string(argv[i + 1]) == "gc_color" ||
                string(argv[i + 1]) == "gc_colorgrad" ||
                string(argv[i + 1]) == "dp_color" ||
                string(argv[i + 1]) == "dp_colorgrad")
                seam_find_type = argv[i + 1];
            else
            {
                cout << "Bad seam finding method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--blend")
        {
            if (string(argv[i + 1]) == "no")
                blend_type = Blender::NO;
            else if (string(argv[i + 1]) == "feather")
                blend_type = Blender::FEATHER;
            else if (string(argv[i + 1]) == "multiband")
                blend_type = Blender::MULTI_BAND;
            else
            {
                cout << "Bad blending method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--blend_strength")
        {
            blend_strength = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--output")
        {
            result_name = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--skip_frames")
        {
            frame_skip = static_cast<int>(atoi(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--mem_limit")
        {
            MAXMEM_MB = static_cast<long>(atol(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--skip_to_segment")
        {
            skip_to_segment = static_cast<int>(atoi(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--force")
        {
            force = true; 
        }
        else if (string(argv[i]) == "--all_frames")
        {
            all_frames = true;
        }
        else if (string(argv[i]) == "--skip_ba")
        {
            skip_ba = true;
        }
        else if (string(argv[i]) == "--skip_reduce_frames")
        {
            reduce_frames = false;
        }
        else if (string(argv[i]) == "--just_show_segments")
        {
            just_show_segments = true;
        }
        else if (video_name.empty())  //only expect one input video file
        {
            //img_names.push_back(argv[i]);
            video_name = argv[i];
            VideoCapture cap(argv[i]);
            if (!cap.isOpened()){
                cout << "Can't open video file '" << argv[i] << "'\n";
                return -1;
            }
            cap.release();
        }
            
    }
    if (preview)
    {
        compose_megapix = 0.6;
    }
    return 0;
}



vector<vector<int>> FindConnectedComponents(std::vector<ImageFeatures> &features,  std::vector<MatchesInfo> &pairwise_matches,
                                      float conf_threshold)
{
    const int num_images = static_cast<int>(features.size());

    DisjointSets comps(num_images);
    for (int i = 0; i < num_images; ++i)
    {
        for (int j = 0; j < num_images; ++j)
        {
            if (pairwise_matches[i*num_images + j].confidence < conf_threshold)
                continue;
            int comp1 = comps.findSetByElem(i);
            int comp2 = comps.findSetByElem(j);
            if (comp1 != comp2)
                comps.mergeSets(comp1, comp2);
        }
    }

    vector<vector<int>> connected_components;
    for (int i = 0; i < num_images; i++){
        vector<int> frames_in_comp;
        for (int j = 0; j < num_images; j++){
            if (comps.findSetByElem(j) == i)
                frames_in_comp.push_back(j);
        }
        if (frames_in_comp.size() > 0)
            connected_components.push_back(frames_in_comp);
    }

    return connected_components;
}



template<typename T>
vector<T> SubVector(vector<T> vec , vector<int> idxs){
    vector<T> subvector;
    subvector.resize(idxs.size());
    for (int i = 0; i < idxs.size(); i++){
        subvector[i] = vec[idxs[i]];
    }

    return subvector;
}


vector<MatchesInfo> GetPairwiseMatchesSubset(vector<MatchesInfo> pairwise_matches , vector<int> idxs){
    vector<MatchesInfo> subset;
    for (int i = 0; i < pairwise_matches.size() ; i++){
        int src_idx = pairwise_matches[i].src_img_idx;
        int dst_idx = pairwise_matches[i].dst_img_idx; 

        int new_src_idx = find(idxs.begin(), idxs.end(), src_idx) - idxs.begin();
        int new_dst_idx = find(idxs.begin(), idxs.end(), dst_idx) - idxs.begin();
        //if (idxs.contains(src_idx) && idxs.contains(dst_idx)) //.constains may not be real, fix this
        //if (find(idxs.begin(), idxs.end(), src_idx) != idxs.end() && find(idxs.begin(), idxs.end(), dst_idx) != idxs.end())
        if (new_src_idx < idxs.size() && new_dst_idx < idxs.size()){
            pairwise_matches[i].src_img_idx = new_src_idx; 
            pairwise_matches[i].dst_img_idx = new_dst_idx; 
            subset.push_back(pairwise_matches[i]); //if there seems to be a problem, maybe I need to do a .clone() similar op here
        }
    }
    return subset;
}



void setmemlimit(long MAXMEM_MB) {
    struct rlimit memlimit;
    long bytes;

    bytes = MAXMEM_MB*(1024*1024);
    memlimit.rlim_cur = bytes;
    memlimit.rlim_max = bytes;
    setrlimit(RLIMIT_AS, &memlimit);
}


double contrast_measure(const Mat&img)
{
    Mat dx,dy, img_copy;
    img.convertTo(img_copy, CV_32F);
    Sobel(img_copy,dx,-1,1,0,CV_32F);
    Sobel(img_copy,dy,-1,0,1,CV_32F);
    magnitude(dx,dy,dx);
    return sum(dx)[0];
}

double CalculateFieldOfView(vector<ImageFeatures> & features , vector<MatchesInfo> & pairwise_matches){

    //jchaves
    //Calculate Homography from feature matches, compare homography transformation to matched feature location

    double max_feature_diff = 0;
    for(int j = 0; j < pairwise_matches.size(); j++){
        vector<DMatch> dmatches = pairwise_matches[j].matches; //vector<DMatch>
        int src_idx = pairwise_matches[j].src_img_idx;
        int dst_idx = pairwise_matches[j].dst_img_idx;
        //cout << j << " out of " << pairwise_matches.size() << " matches.  Frames " << src_idx << " , " << dst_idx << endl;
        if (src_idx < 0 || dst_idx < 0) continue;  //was getting OutOfMemory errors w/o this

        vector<Point2f> img1_points; //will be in order so that points are of corresponding matched features
        vector<Point2f> img2_points;
      
        if (dmatches.size() == 0) continue; //happens sometimes, for some reason, was causing error in findHomography 
        for(int m = 0; m < dmatches.size(); m++){

            int i1 = dmatches[m].queryIdx;
            int i2 = dmatches[m].trainIdx;
            CV_Assert(i1 >= 0 && i1 < static_cast<int>(features[src_idx].keypoints.size()));
            CV_Assert(i2 >= 0 && i2 < static_cast<int>(features[dst_idx].keypoints.size()));

            img1_points.push_back(features[src_idx].keypoints[i1].pt);
            img2_points.push_back(features[dst_idx].keypoints[i2].pt);

        }

        //Now I have vectors of corresponding matched feature points
        //Compute Homography
        Mat H = findHomography(img1_points, img2_points, CV_RANSAC);
        if (H.data == 0x0) continue;  //H not calculated properly, could be because not enough matching features to estimate it

        //Mat H_other = pairwise_matches[j].H;  //didn't realize that pairwise_matches already had an estimated homography

        for(int m = 0; m < img1_points.size(); m++){
            //Mat img1 = feature_frames[src_idx].clone();
            //Mat img2 = feature_frames[dst_idx].clone();

            //circle(img1, img1_points[m], 10, Scalar(0,0,255));
            //circle(img2, img2_points[m], 10, Scalar(0,255,0));
            double point_norm = (H.at<double>(2,0)*img1_points[m].x + H.at<double>(2,1)*img1_points[m].y + H.at<double>(2,2));
            Point2f homography_point = Point2f(
                    (H.at<double>(0,0)*img1_points[m].x + H.at<double>(0,1)*img1_points[m].y + H.at<double>(0,2))/point_norm , 
                    (H.at<double>(1,0)*img1_points[m].x + H.at<double>(1,1)*img1_points[m].y + H.at<double>(1,2))/point_norm
                    );
            //circle(img2, homography_point, 8, Scalar(255,0,0));  //make this a bit smaller, so if they're on top of each other, you can see both circles
            //Try plotting the homography-xformed point both w/ and w/o normalizing the 3rd homogeneous coord of the output
            //ie dividing the new x any by (H.at<double>(2,0)*img1_points[m].x + H.at<double>(2,1)*img1_points[m].y + H.at<double>(2,2))

            //cout << "Homography H = " << endl << H << endl;
            //cout << "first row = " << H.at<double>(0,0) << " " << H.at<double>(0,1) << " " << H.at<double>(0,2) << endl;

            Point2f feature_difference = img2_points[m] - img1_points[m];
            double feature_diff = sqrt(feature_difference.x*feature_difference.x  +  feature_difference.y*feature_difference.y);
           

            Point2f homography_difference = img2_points[m] - homography_point;
            double homography_diff = sqrt(homography_difference.x*homography_difference.x  +  homography_difference.y*homography_difference.y);
            if (homography_diff < 5)
                if (feature_diff > max_feature_diff)
                    max_feature_diff = feature_diff;

        }


        //waitKey(); //needs user input , don't need if break stmt above
    }

     
    return max_feature_diff;

}


//Functor for Sorting by overlap
struct by_overlap {
    bool operator()(tuple<int,int,float> const & a, tuple<int,int,float> const & b) const {
        return std::get<2>(a) > std::get<2>(b); // use > so that std::sort gives descending ordered vector
    }
};


void ReduceFramesByOverlap(vector<Mat> & frames, vector<Mat> & feature_frames , vector<ImageFeatures> & features , vector<MatchesInfo> & pairwise_matches){

    tuple<int,int,float> empty_tuple (0,0, 0.);
    vector<tuple<int,int,float>> ordered_overlaps (pairwise_matches.size() , empty_tuple);

    //Also consider how well each frame matches to other frames
    //as a metric on top of blurriness to consider when deciding
    //which frame should be removed
    vector<int> num_frames_matched (frames.size(), 0);



    //namedWindow("First image");
    //namedWindow("Second image");
    //namedWindow("Intersection");

    for(int j = 0; j < pairwise_matches.size(); j++){
        vector<DMatch> dmatches = pairwise_matches[j].matches; //vector<DMatch>
        int src_idx = pairwise_matches[j].src_img_idx;
        int dst_idx = pairwise_matches[j].dst_img_idx;
        //cout << j << " out of " << pairwise_matches.size() << " matches.  Frames " << src_idx << " , " << dst_idx << endl;
        if (src_idx < 0 || dst_idx < 0) continue;  //was getting OutOfMemory errors w/o this

        vector<Point2f> img1_points; //will be in order so that points are of corresponding matched features
        vector<Point2f> img2_points;
      
        if (dmatches.size() == 0) continue; //happens sometimes, for some reason, was causing error in findHomography 
        for(int m = 0; m < dmatches.size(); m++){

            int i1 = dmatches[m].queryIdx;
            int i2 = dmatches[m].trainIdx;
            CV_Assert(i1 >= 0 && i1 < static_cast<int>(features[src_idx].keypoints.size()));
            CV_Assert(i2 >= 0 && i2 < static_cast<int>(features[dst_idx].keypoints.size()));

            img1_points.push_back(features[src_idx].keypoints[i1].pt);
            img2_points.push_back(features[dst_idx].keypoints[i2].pt);

        }

        //Now I have vectors of corresponding matched feature points
        //Compute Homography
        Mat H = findHomography(img1_points, img2_points, CV_RANSAC);
        if (H.data == 0x0) continue;  //H not calculated properly, could be because not enough matching features to estimate it
 


        //warpPerspective test
        Mat warp_output;
        Mat im1 = feature_frames[src_idx]; // don't need .clone() b/c not changing it, just want to refer to frame
        Mat im2 = feature_frames[dst_idx];
        Mat ones_mask = Mat::ones(im1.rows, im1.cols, CV_8U);
        Mat warp_input;
        Mat ones_out1;
        Mat ones_out2;
        Mat translation = H.clone(); 
        translation.at<double>(0,0) = 1; translation.at<double>(0,1) = 0; 
        translation.at<double>(1,0) = 0; translation.at<double>(1,1) = 1;
        translation.at<double>(2,0) = 0; translation.at<double>(2,1) = 0;  
        translation.at<double>(0,2) = im1.cols/2; translation.at<double>(1,2) = im1.rows/2;  // may be switched
        warpPerspective(im1 , warp_input , H*translation ,  Size(im1.cols*2 , im1.rows*2));
        warpPerspective(im2 , warp_output , translation , Size(im1.cols*2 , im1.rows*2));
        //Remember: H maps from frame 1 to frame 2, so I should compare the images of H*im1 and im2, or im1 and H^-1 * im2
        warpPerspective(ones_mask , ones_out1 , translation ,  Size(im1.cols*2 , im1.rows*2));
        warpPerspective(ones_mask , ones_out2 , H*translation , Size(im1.cols*2 , im1.rows*2));
        Mat intersection;
        bitwise_and(ones_out1 , ones_out2 , intersection);
        //imshow("First image" , warp_input);
        //imshow("Second image" , warp_output);
        //imshow("Intersection" , intersection);
        long intersection_area = (sum(intersection))[0];
        long max_int_area = static_cast<long>( std::min((sum(ones_out1))[0] , (sum(ones_out2))[0]) );
        //cout << src_idx << ", " << dst_idx << ": " << static_cast<float>(intersection_area)/static_cast<float>(max_int_area) << " " << contrast_measure(im1) << " " << contrast_measure(im2) << endl;
        //waitKey(10*1000);
        ///////////////////////////////////
        
        tuple<int,int,float> entry (src_idx, dst_idx, static_cast<float>(intersection_area)/static_cast<float>(max_int_area));
        ordered_overlaps[j] = entry;
        num_frames_matched[src_idx]++;
        num_frames_matched[dst_idx]++;

    }

    std::sort(ordered_overlaps.begin() , ordered_overlaps.end() , by_overlap());


    float min_overlap = 0.99; //i.e. 99% of area
    vector<bool> remove_frames (feature_frames.size() , false);

    for (int i = 0; i < ordered_overlaps.size(); i++){
        if (std::get<2>(ordered_overlaps[i]) < min_overlap) break; //recall, vector is sorted in descending order
        int idx1 = std::get<0>(ordered_overlaps[i]);
        int idx2 = std::get<1>(ordered_overlaps[i]);

        if (!remove_frames[idx1] && !remove_frames[idx2]){//I.e. if both frames have not already been set to be removed
            
            //float score = (contrast_measure(feature_frames[idx1]) - contrast_measure(feature_frames[idx2]))/1e8 + (num_frames_matched[idx1] - num_frames_matched[idx2])/std::min(num_frames_matched[idx1], num_frames_matched[idx2]);
            float score = (num_frames_matched[idx1] - num_frames_matched[idx2])/std::min(num_frames_matched[idx1], num_frames_matched[idx2]);
            //int remove_idx = ( contrast_measure(feature_frames[idx1]) < contrast_measure(feature_frames[idx2]) ) ? idx1 : idx2 ;
            int remove_idx = (score < 0) ? idx1 : idx2;
            remove_frames[remove_idx] = true;

        }

    }

    //Now I've identified which frames I want to remove
    //I can go ahead and remove them from feature_frames and features and frames
    //Test manually getting subset of pairwise_matches with my function above
    //I can try adding a .clone() type call in that function like mentioned if need be
    //No need for multiple passes when full N^2 matching is sent into this function
    

    for (int i = frames.size() - 1 ; i >= 0; i--){ //need to go backwards when removing elements to keep indices the same
        if (remove_frames[i]){
            frames.erase(frames.begin() + i);
            feature_frames.erase(feature_frames.begin() + i);
            features.erase(features.begin() + i);
        }
    }

    for (int i = 0; i < features.size(); i++) { //now reduced size
        features[i].img_idx = i;
    }

    //cout << "Remaining # of Frames: " << frames.size() << endl;
}







// Function Prototypes
void TryToStitchVideoSegment(vector<Mat> frames , vector<Mat> feature_frames , vector<ImageFeatures> features , vector<MatchesInfo> pairwise_matches, int segment_id);
Mat CreatePanorama(vector<Mat> & frames , vector<ImageFeatures> & features , vector<MatchesInfo> & pairwise_matches);





int main(int argc, char* argv[])
{
#if ENABLE_LOG
    int64 app_start_time = getTickCount();
#endif

    //cv::setBreakOnError(true);

    int retval = parseCmdArgs(argc, argv);
    if (retval)
        return retval;


    setmemlimit(MAXMEM_MB);


    VideoCapture cap(video_name);
    Mat frame;
    vector<double> frame_contrasts;
    for(int i = 0; i < cap.get(CV_CAP_PROP_FRAME_COUNT) ; i++) { 
        cap >> frame;
        if (!frame.empty() && i%frame_skip == 0){
            original_frames.push_back(frame.clone());
            //img_names.push_back("frame " + to_string(i));  //still good for printing match connection graph
            frame_contrasts.push_back(contrast_measure(frame));
        }
    
        //I should check the frames here or in main(), to make sure they're valid frames, ie 
        //use countNonZeros from OpenCV to check Mat and also could do an early cull of frames with poor sharpness or other
        //quality metric
    
    
    }
    cap.release();

    ////Cull images of excessively poor quality
    //double sum = accumulate(frame_contrasts.begin(), frame_contrasts.end(), 0.0);
    //double mean = sum / frame_contrasts.size();

    //vector<double> diff(frame_contrasts.size());
    //std::transform(frame_contrasts.begin(), frame_contrasts.end(), diff.begin(),
    //                       std::bind2nd(std::minus<double>(), mean));
    //double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    //double stdev = std::sqrt(sq_sum / frame_contrasts.size());

    //for (int i = frame_contrasts.size() - 1; i >= 0; i--){
    //    if (frame_contrasts[i] < mean - 2*stdev)
    //        original_frames.erase(original_frames.begin() + i);
    //}
    frame_contrasts.clear();
    for (int i = 0; i < original_frames.size(); i++)
        img_names.push_back("frame " + to_string(i));



    // Check if have enough images
    int num_images = static_cast<int>(img_names.size());
    if (num_images < 2)
    {
        LOGLN("Need more images");
        return -1;
    }

    double work_scale = 1, seam_scale = 1, compose_scale = 1;
    bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

    LOGLN("Finding features...");
#if ENABLE_LOG
    int64 t = getTickCount();
#endif

    Ptr<FeaturesFinder> finder;
    if (features_type == "surf")
    {
#ifdef HAVE_OPENCV_NONFREE
        if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
            finder = makePtr<SurfFeaturesFinderGpu>();
        else
#endif
            finder = makePtr<SurfFeaturesFinder>();
    }
    else if (features_type == "orb")
    {
        finder = makePtr<OrbFeaturesFinder>();
    }
    else
    {
        cout << "Unknown 2D features type: '" << features_type << "'.\n";
        return -1;
    }

    Mat full_img, img;
    vector<ImageFeatures> features(num_images);
    vector<Mat> images(num_images);
    vector<Size> full_img_sizes(num_images);
    double seam_work_aspect = 1;

    vector<Mat> feature_frames; //Added by JChaves, used for Homography-based motion measurement, shouldn't be a global

    //VideoCapture cap(video_name);
    namedWindow("Dispay");
    for (int i = 0; i < num_images; ++i)
    {
        //full_img = imread(img_names[i]);
        //cap >> full_img;
        full_img = original_frames[i].clone();
        //Mat temp = full_img.clone(); //without this, after this loop terminated, all elements of original_frames were just the last frame
        //original_frames.push_back(temp);
        //imshow("Display", original_frames[original_frames.size() - 1]);
        //waitKey(100);
        full_img_sizes[i] = full_img.size();

        if (full_img.empty())
        {
            LOGLN("Can't open image " << img_names[i]);
            return -1;
        }
        if (work_megapix < 0)
        {
            img = full_img;
            work_scale = 1;
            is_work_scale_set = true;
        }
        else
        {
            if (!is_work_scale_set)
            {
                work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
                is_work_scale_set = true;
            }
            resize(full_img, img, Size(), work_scale, work_scale); //resize full_img to img to input into FeatureFinder fn
        }
        if (!is_seam_scale_set)
        {
            seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }

        (*finder)(img, features[i]);
        features[i].img_idx = i;
        //LOGLN("Features in image #" << i+1 << ": " << features[i].keypoints.size());

        feature_frames.push_back(img.clone());

        //Show feature keypoints on images
        //for(int j = 0; j < features[i].keypoints.size(); j++){
        //    circle(img, features[i].keypoints[j].pt, 10, Scalar(0,0,255));
        //}

        //imshow("Display", img);
        //waitKey(500);

        resize(full_img, img, Size(), seam_scale, seam_scale);  //resize full_img to img to be copied into images and later be used for seam work
        images[i] = img.clone();  //note, the .clone() is important, it makes the assignment a deep copy, rather than a shallow, referenced assignment
    }
    //cap.release();

    //Confirm that original_frames elements are unique and correct
    //for(int i = 0; i < original_frames.size(); i++){
    //    imshow("Display", original_frames[i]);
    //    waitKey(100);
    //}
    //imshow("Display" , original_frames[219]);
    //waitKey();
    

    finder->collectGarbage();
    full_img.release();
    img.release();

    LOGLN("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    LOGLN("Pairwise matching...");
#if ENABLE_LOG
    t = getTickCount();
#endif
    vector<MatchesInfo> pairwise_matches;
    BestOf2NearestMatcher matcher(try_cuda, match_conf);
    Mat matchMask(features.size(),features.size(),CV_8U,Scalar(0));
    int match_window_size = 10;
    for (int i = 0; i < num_images; ++i){
        for (int j = -match_window_size ; j <= match_window_size ; j++){
            int idx = i + j;
            if (idx >= 0 && idx < num_images)
                matchMask.at<char>(i , idx) = 1;

        }
    }


    matcher(features, pairwise_matches, matchMask);
    matcher.collectGarbage();
    LOGLN("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");



    vector<vector<int>> connected_components;
    if (!all_frames){
        ///// Here, after all the pairwise matching has been done on all the frames in the video, I want to 
        ///// break the video up into segments that can individually be used to attempt to make separate panoramas
        // Will first try a connected component technique, similar to leaveBiggestComponent(features, pairwise_matches, conf_thresh)

        connected_components = FindConnectedComponents(features, pairwise_matches, conf_thresh);   
        // Will likely need to find a good value of conf_thresh to make good separation of video segments
        // Just like match_conf was adjusted to a value of 0.9 to really enforce good matches, which help to enforce
        // image quality and camera motion constraints
        //// May need to use Structure From Motion analysis to separate panning motion parts of video. 

        namedWindow("Segments");
        for (int i = 0; i < connected_components.size(); i++) {
            if (connected_components[i].size() > min_segment_size)
                cout << "New Segment: ";
            else
                cout << "Excluded Frames: ";
            for (int j = 0; j < connected_components[i].size(); j++) {
                cout << connected_components[i][j] << " ";//"(" << contrast_measure(original_frames[connected_components[i][j]]) << ") ";
                imshow("Segments" , original_frames[connected_components[i][j]]);
                waitKey(60);
            }
            cout << endl;
            waitKey(500);
        }
        destroyWindow("Segments");
    }
    
    if (just_show_segments)
        return 0;

    if (all_frames){
        TryToStitchVideoSegment(original_frames , feature_frames , features, pairwise_matches, 0);
    }else{

        if (skip_to_segment == 0) {
            for (int i = 0; i < connected_components.size(); i++){
                TryToStitchVideoSegment(SubVector<Mat>(original_frames , connected_components[i]) , 
                                        SubVector<Mat>(feature_frames , connected_components[i]) ,
                                        SubVector<ImageFeatures>(features, connected_components[i]) , 
                                        GetPairwiseMatchesSubset(pairwise_matches , connected_components[i]) , i);
            }
            
        }else{
           int seg_counter = 0;
            for (int i = 0; i < connected_components.size(); i++){
                if (connected_components[i].size() > min_segment_size){
                    seg_counter++;
                    if (seg_counter == skip_to_segment){
                        result_name_counter = seg_counter;
                        TryToStitchVideoSegment(SubVector<Mat>(original_frames , connected_components[i]) , 
                                        SubVector<Mat>(feature_frames , connected_components[i]) ,
                                        SubVector<ImageFeatures>(features, connected_components[i]) , 
                                        GetPairwiseMatchesSubset(pairwise_matches , connected_components[i]) , i);
                        break;
                    }
                }
            }

        }

    }


    LOGLN("Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");


    return 0;
} // end of main fn (under my reorganization)



void TryToStitchVideoSegment(vector<Mat> frames , vector<Mat> feature_frames , vector<ImageFeatures> features , vector<MatchesInfo> pairwise_matches , int segment_id){ //Do NOT take in argument reference (ie &) because I will be changing frames in this fn
    //Here, I have one video segment that I want to make into a panorama
    //But, I want to do the stitching in a partitioned way if there are too many frames to start
  
    int64 t;

    if (reduce_frames){
        //This will reduce frames based on windowed matching info, but the reduction should be faster
        //Might as well make the reduction less aggressive too, made min_overlap 0.99
        //This will greatly speed up the N^2 matching for FoV calculation and the rest of the process
        cout << "Reducing # of frames based on overlap" << endl;
        cout << "Original Number of Frames: " << frames.size() << endl;
        t = getTickCount();
        ReduceFramesByOverlap(frames, feature_frames , features, pairwise_matches);
        cout << "Reduced Number of Frames: " << frames.size() << endl << endl;
        cout << "Frame Reduction, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;
        //cout << "Done." << endl << endl;
        namedWindow("Reduced Frames");
        for (int i = 0; i < frames.size(); i++){
            imshow("Reduced Frames", frames[i]);
            waitKey(500);
        }
        destroyWindow("Reduced Frames");
        
        //If there's a problem with not many of the reduced frames being in the same
        //connected component, ie all matched, when making the panorama
        //You could try increasing the --work_megapix param, to increase features found
    }





    // I would also like to check that this whole video segment has a large enough field-of-view to be
    // worthy or being made into a panorama
    // For that I will need feature matching for just this video segment
    // I can try to use the matched feature image coordinate distance idea I had 

    //vector<MatchesInfo> pairwise_matches;
    cout << "Re-matching Segment (possibly after Frame Reduction)..." << endl;
    t = getTickCount();
    pairwise_matches.clear();  //Since I actually just re-calculate it later in CreatePanorama()
    BestOf2NearestMatcher matcher(try_cuda, match_conf);
    matcher(features, pairwise_matches);  //Need to do full N^2 matching for field-of-view
    matcher.collectGarbage();
    cout << "Re-matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;

    //Go through each matched feature, and find the max difference in image position
    //for a pair of matched features whose homography error is less than 5 or 10 pixels
    if (!force) {
        cout << "Checking Field-of-View..." << endl; 
        t = getTickCount();


        double field_of_view = CalculateFieldOfView(features, pairwise_matches);
        cout << "Field-of-View Check, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;

        //Uncomment the following 2 lines to just see the field of view measure for the different segments
        //cout << "field of view measure = " << field_of_view << endl;
        //return;

        double fov_thresh = 30; // pixels.  Maybe make this a global?

        if (field_of_view < fov_thresh){
            cout << "Segment " << segment_id << " does not have a large enough field-of-view" << endl;
            cout << "Panorama creation skipped." << endl << endl;
         
            return;  //not enough field-of-view, skip this segment
                     //maybe I should still increment result_name_counter?
        }

    }




    Mat result;
    int num_frames = frames.size();
    //The constant numbers below are subject to change based on testing this program with long videos
    if (num_frames > 8000000){ 
        //vector<Mat> new_frames;
        //while (frames.size() > 40) { //Do this partitioned calling of CreatePanorama until the total number of imgs left is < 40
        //    int times = num_frames/40;  // integer truncation used here
        //    //new_frames.resize(times);
        //    for (int i = 0; i < times; i++){
        //        vector<Mat> input_frames(frames.begin() + i*40 , frames.begin() + (i+1)*40 );
        //        result = CreatePanorama(input_frames);
        //        if (!result.empty()) new_frames.push_back(result.clone());
        //        input_frames.clear();
        //    }
        //    if (num_frames % 40 > 2){
        //        vector<Mat> input_frames(frames.begin() + times*40 , frames.end());
        //        result = CreatePanorama(input_frames);
        //        if (!result.empty()) new_frames.push_back(result.clone());
        //        input_frames.clear();
        //    }
        //        
        //    frames = new_frames;
        //    new_frames.clear();
        //}

        //result = CreatePanorama(frames);

    }else if (num_frames > min_segment_size){ //Don't bother creating a panorama unless the segment has > 5 frames
        result = CreatePanorama(frames, features, pairwise_matches);
    }

    if (!result.empty())
        imwrite(result_name + to_string(result_name_counter++) + ".jpg", result);

}


Mat CreatePanorama(vector<Mat> & frames , vector<ImageFeatures> & features , vector<MatchesInfo> & pairwise_matches){



    int num_images = static_cast<int>(frames.size());
    if (num_images < 2)
    {
        LOGLN("Need more images");
        return Mat();
    }

    // Need to make a temp local version of this for code below
    // Overrides the global img_names which is needed in parse_args and main
    vector<String> img_names;
    img_names.resize(num_images);
    for(int i = 0; i < num_images; i++)
        img_names[i] = "Temp Frame " + to_string(i);




    double work_scale = 1, seam_scale = 1, compose_scale = 1;
    bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

    //LOGLN("Finding features...");
//#if //ENABLE_LOG
    int64 t = getTickCount();
//#end//if

    //Ptr<FeaturesFinder> finder;
    //if (features_type == "surf")
    //{
//#ifd//ef HAVE_OPENCV_NONFREE
    //    if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
    //        finder = makePtr<SurfFeaturesFinderGpu>();
    //    else
//#end//if
    //        finder = makePtr<SurfFeaturesFinder>();
    //}
    //else if (features_type == "orb")
    //{
    //    finder = makePtr<OrbFeaturesFinder>();
    //}
    //else
    //{
    //    cout << "Unknown 2D features type: '" << features_type << "'.\n";
    //    return Mat();
    //}

    Mat full_img, img;
    //vector<ImageFeatures> features(num_images);
    vector<Mat> images(num_images);
    vector<Size> full_img_sizes(num_images);
    double seam_work_aspect = 1;

    vector<Mat> feature_frames; //Added by JChaves, used for Homography-based motion measurement, shouldn't be a global

    //VideoCapture cap(video_name);
    namedWindow("Dispay");
    for (int i = 0; i < num_images; ++i)
    {
        //full_img = imread(img_names[i]);
        //cap >> full_img;
        full_img = frames[i].clone();
        //Mat temp = full_img.clone(); //without this, after this loop terminated, all elements of original_frames were just the last frame
        //original_frames.push_back(temp);
        //imshow("Display", original_frames[original_frames.size() - 1]);
        imshow("Display" , frames[i]);
        waitKey(100);
        full_img_sizes[i] = full_img.size();

        if (full_img.empty())
        {
            LOGLN("Can't open image " << i); //img_names[i]);
            return Mat();
        }
        if (work_megapix < 0)
        {
            img = full_img;
            work_scale = 1;
            is_work_scale_set = true;
        }
        else
        {
            if (!is_work_scale_set)
            {
                work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
                is_work_scale_set = true;
            }
            resize(full_img, img, Size(), work_scale, work_scale); //resize full_img to img to input into FeatureFinder fn
        }
        if (!is_seam_scale_set)
        {
            seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }

        //(*finder)(img, features[i]);

        //even though I'm not re-finding the features, this line is important for later parts
        features[i].img_idx = i; 

        //LOGLN("Features in image #" << i+1 << ": " << features[i].keypoints.size());

        feature_frames.push_back(img.clone());

        //Show feature keypoints on images
        //for(int j = 0; j < features[i].keypoints.size(); j++){
        //    circle(img, features[i].keypoints[j].pt, 10, Scalar(0,0,255));
        //}

        //imshow("Display", img);
        //waitKey(500);

        resize(full_img, img, Size(), seam_scale, seam_scale);  //resize full_img to img to be copied into images and later be used for seam work
        images[i] = img.clone();  //note, the .clone() is important, it makes the assignment a deep copy, rather than a shallow, referenced assignment
    }
    //cap.release();

    //Confirm that original_frames elements are unique and correct
    //for(int i = 0; i < original_frames.size(); i++){
    //    imshow("Display", original_frames[i]);
    //    waitKey(100);
    //}

    //finder->collectGarbage();
    full_img.release();
    img.release();

    //LOGLN("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");








    //Lower thresholds for reduced frame segment
    //No, produces bad results
    //float new_conf_thresh = 0.01;





    //LOG("Pairwise matching");
//#if //ENABLE_LOG
    //t = getTickCount();
//#end//if
    //vector<MatchesInfo> pairwise_matches;
    //BestOf2NearestMatcher matcher(try_cuda, match_conf);
    //matcher(features, pairwise_matches);
    //matcher.collectGarbage();
    //LOGLN("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    //vector<MatchesInfo> pairwise_matches;
    //pairwise_matches.clear();
    //BestOf2NearestMatcher matcher(try_cuda, match_conf);
    //Mat matchMask(features.size(),features.size(),CV_8U,Scalar(0));
    //int match_window_size = 10;
    //for (int i = 0; i < num_images; ++i){
    //    for (int j = -match_window_size ; j <= match_window_size ; j++){
    //        int idx = i + j;
    //        if (idx >= 0 && idx < num_images)
    //            matchMask.at<char>(i , idx) = 1;

    //    }
    //}


    //matcher(features, pairwise_matches, matchMask);
    //matcher(features, pairwise_matches);
    //matcher.collectGarbage();






















    if (!force) {
        cout << "Checking Camera Motion Model...";
        //jchaves
        //Calculate Homography from feature matches, compare homography transformation to matched feature location
        int num_points_used = 0;
        double average_point_error = 0;
        namedWindow("First image");
        namedWindow("Second image");
        namedWindow("Intersection");
        for(int j = 0; j < pairwise_matches.size(); j++){
            vector<DMatch> dmatches = pairwise_matches[j].matches; //vector<DMatch>
            int src_idx = pairwise_matches[j].src_img_idx;
            int dst_idx = pairwise_matches[j].dst_img_idx;
            //cout << j << " out of " << pairwise_matches.size() << " matches.  Frames " << src_idx << " , " << dst_idx << endl;
            if (src_idx < 0 || dst_idx < 0) continue;  //was getting OutOfMemory errors w/o this

            vector<Point2f> img1_points; //will be in order so that points are of corresponding matched features
            vector<Point2f> img2_points;
          
            if (dmatches.size() == 0) continue; //happens sometimes, for some reason, was causing error in findHomography 
            for(int m = 0; m < dmatches.size(); m++){

                int i1 = dmatches[m].queryIdx;
                int i2 = dmatches[m].trainIdx;
                CV_Assert(i1 >= 0 && i1 < static_cast<int>(features[src_idx].keypoints.size()));
                CV_Assert(i2 >= 0 && i2 < static_cast<int>(features[dst_idx].keypoints.size()));

                img1_points.push_back(features[src_idx].keypoints[i1].pt);
                img2_points.push_back(features[dst_idx].keypoints[i2].pt);

            }

            //Now I have vectors of corresponding matched feature points
            //Compute Homography
            Mat H = findHomography(img1_points, img2_points, CV_RANSAC);
            if (H.data == 0x0) continue;  //H not calculated properly, could be because not enough matching features to estimate it

            //Mat H_other = pairwise_matches[j].H;  //didn't realize that pairwise_matches already had an estimated homography


            for(int m = 0; m < img1_points.size(); m++){
                Mat img1 = feature_frames[src_idx].clone();
                Mat img2 = feature_frames[dst_idx].clone();

                //circle(img1, img1_points[m], 10, Scalar(0,0,255));
                //circle(img2, img2_points[m], 10, Scalar(0,255,0));
                double point_norm = (H.at<double>(2,0)*img1_points[m].x + H.at<double>(2,1)*img1_points[m].y + H.at<double>(2,2));
                Point2f homography_point = Point2f(
                        (H.at<double>(0,0)*img1_points[m].x + H.at<double>(0,1)*img1_points[m].y + H.at<double>(0,2))/point_norm , 
                        (H.at<double>(1,0)*img1_points[m].x + H.at<double>(1,1)*img1_points[m].y + H.at<double>(1,2))/point_norm
                        );
                //circle(img2, homography_point, 8, Scalar(255,0,0));  //make this a bit smaller, so if they're on top of each other, you can see both circles
                //Try plotting the homography-xformed point both w/ and w/o normalizing the 3rd homogeneous coord of the output
                //ie dividing the new x any by (H.at<double>(2,0)*img1_points[m].x + H.at<double>(2,1)*img1_points[m].y + H.at<double>(2,2))

                //cout << "Homography H = " << endl << H << endl;
                //cout << "first row = " << H.at<double>(0,0) << " " << H.at<double>(0,1) << " " << H.at<double>(0,2) << endl;


                Point2f difference = img2_points[m] - homography_point;
                double diff = sqrt(difference.x*difference.x  +  difference.y*difference.y);
                //if (diff > 20)
                //    diff = 20;
                average_point_error += diff ; //* dmatches[m].distance;
                //cout << sqrt(difference.x*difference.x  +  difference.y*difference.y) << "   " << dmatches[m].distance << endl;
                //cout << pairwise_matches[j].confidence << endl;
                num_points_used++;

                //imshow("First image", img1);
                //imshow("Second image", img2);

                //int c;
                //if (sqrt(difference.x*difference.x  +  difference.y*difference.y) > 5)
                //    c = waitKey(50000);
                //else
                //    c = waitKey(1);
                //
                //if (c == 1048603) break;  // ESC key (for some reason isn't ASCII 27) to quit looking at this frame pair and go to the next pair


            }


            //waitKey(); //needs user input , don't need if break stmt above

        }

        average_point_error /= num_points_used;
        cout << endl << "The average point error is " << average_point_error << endl;
        if (average_point_error > 0.5) {
            cout << "A segment was found to have improper camera motion for stitching" << endl;
            return Mat();
        }
        
        
        //cout << "Done." << endl;
        ////////////////////////////////////////
    }
    


    //Display Feature Matches
    //Mat outTemp;
    //for(int j = 0; j < pairwise_matches.size(); j++){
    //    int src_idx = pairwise_matches[j].src_img_idx;
    //    int dst_idx = pairwise_matches[j].dst_img_idx;
    //    if (src_idx < 0 || dst_idx < 0) continue;  //was getting OutOfMemory errors w/o this
    //    drawMatches(feature_frames[src_idx], features[src_idx].keypoints, feature_frames[dst_idx], features[dst_idx].keypoints, pairwise_matches[j].matches, outTemp);
    //    imshow("Display", outTemp);
    //    waitKey(800);    
    //}


    // Check if we should save matches graph
    //if (save_graph)
    //{
    //    LOGLN("Saving matches graph...");
    //    ofstream f(save_graph_to.c_str());
    //    f << matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
    //}


    vector<int> indices;
    if (all_frames) {
        conf_thresh = 0;
    }
        // Leave only images we are sure are from the same panorama
        indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);  //presumably selects the largest connected component from the graph of pairwise matches, ie selects which images should be used to make a panorama based on which set of images have the largest connected graph
        vector<Mat> img_subset;
        vector<String> img_names_subset;
        vector<Size> full_img_sizes_subset;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            img_names_subset.push_back(img_names[indices[i]]);
            img_subset.push_back(images[indices[i]]);
            full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
        }

        cout << "Original number of images: " << frames.size() << endl;
        cout << "Remaining number of images: " << indices.size() << endl;

        images = img_subset;
        img_names = img_names_subset;
        full_img_sizes = full_img_sizes_subset;

        // Check if we still have enough images
        num_images = static_cast<int>(img_names.size());
        if (num_images < 2)
        {
            LOGLN("Need more images");
            return Mat();
        }
    
    //}else{
    //    indices.resize(num_images);
    //    for (int i = 0; i < num_images; i++)
    //        indices[i] = i;
    //}


    HomographyBasedEstimator estimator;
    vector<CameraParams> cameras;
    if (!estimator(features, pairwise_matches, cameras))
    {
        cout << "Homography estimation failed.\n";
        return Mat();
    }

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);  //cameras[i].R gives the Rotation Matrix
        //Translation Matrix .t is assumed to be zero during whole stitching pipeline because 
        //stitching, or maybe just Homography based estimation of camera matrices, requires
        //that the camera only rotate about its optical center, or that the scene photographed is far and planar so as to approximate that
        //May be interesting to look at .t of these camera matrices, if it is filled, to see if the images are good for stitching? 
        cameras[i].R = R;  //In the above convertTo call, the Mat R is the output of the conversion, so set cameras[i].R to it here
        //LOGLN("Initial intrinsics #" << indices[i]+1 << ":\n" << cameras[i].K());
        //LOGLN("Rotation matrix #" << indices[i] + 1 << ":\n" << cameras[i].R);
        //LOGLN("Translation matrix #" << indices[i] + 1 << ":\n" << cameras[i].t); //assumed, and therefore set, to zeros
        //This reports the intrinsic matrix K, which should be independent of the Rotation from the coordinate system origin,
        //so are these together just because the for loop would be the same?
    }

    if (!skip_ba) {
        Ptr<detail::BundleAdjusterBase> adjuster;
        int num_params_per_cam_ , num_errs_per_measurement_;
        if (ba_cost_func == "reproj"){
             adjuster = makePtr<detail::BundleAdjusterReproj>();
             num_params_per_cam_ = 7;
             num_errs_per_measurement_ = 2;
        }else if (ba_cost_func == "ray"){
            adjuster = makePtr<detail::BundleAdjusterRay>();
            num_params_per_cam_ = 4;
            num_errs_per_measurement_ = 3;
        }else{
            cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
            return Mat();
        }
        adjuster->setConfThresh(conf_thresh);
        Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
        if (ba_refine_mask[0] == 'x') refine_mask(0,0) = 1;
        if (ba_refine_mask[1] == 'x') refine_mask(0,1) = 1;
        if (ba_refine_mask[2] == 'x') refine_mask(0,2) = 1;
        if (ba_refine_mask[3] == 'x') refine_mask(1,1) = 1;
        if (ba_refine_mask[4] == 'x') refine_mask(1,2) = 1;
        adjuster->setRefinementMask(refine_mask);
        //Sometimes, can get an out of memory error if too much is given to the bundle adjustment
        //idk if it's based just on the number of frames, or maybe the number of features, or matches, or whatever else
        //I wonder if this can be skipped or something since the video uses the same camera (although maybe more than K is important)
        //The pairwise_matches.size() obv changes significantly with match_conf, so maybe that's what gets too large and cause memory error
        if (!(*adjuster)(features, pairwise_matches, cameras))  //camera parameters adjusted from initial guess by bundle adjustment
        {
            cout << "Camera parameters adjusting failed.\n";
            return Mat();
        }
        //But I thought bundle adjustment was for adjusting the estimation of camera matrices M, not intrinsic camera matrices K
        //Then why is the cameras[i].K() affected???
    }




    // Find median focal length

    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        //LOGLN("Camera #" << indices[i]+1 << ":\n" << cameras[i].K());
        focals.push_back(cameras[i].focal);
    }

    sort(focals.begin(), focals.end());
    float warped_image_scale;  // = median focal length
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;


    // Wave Correction
    // adjusts rotation matrices for each camera to make the panorama more horizontal or vertical, to get a good rectangular panorama
    if (do_wave_correct)
    {
        vector<Mat> rmats;
        for (size_t i = 0; i < cameras.size(); ++i)
            rmats.push_back(cameras[i].R);
        waveCorrect(rmats, wave_correct);
        for (size_t i = 0; i < cameras.size(); ++i)
            cameras[i].R = rmats[i];
    }



    // Warping images
    // Will project images and masks to a specified geometry, e.g plane, unit sphere, unit cylinder, etc

    LOGLN("Warping images (auxiliary)... ");
#if ENABLE_LOG
    t = getTickCount();
#endif

    vector<Point> corners(num_images);
    vector<Mat> masks_warped(num_images);
    vector<Mat> images_warped(num_images);
    vector<Size> sizes(num_images);
    vector<Mat> masks(num_images);

    // Preapre images masks
    for (int i = 0; i < num_images; ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // Warp images and their masks

    Ptr<WarperCreator> warper_creator;
    if (try_ocl)
    {
        if (warp_type == "plane")
            warper_creator = makePtr<cv::PlaneWarperOcl>();
        else if (warp_type == "cylindrical")
            warper_creator = makePtr<cv::CylindricalWarperOcl>();
        else if (warp_type == "spherical")
            warper_creator = makePtr<cv::SphericalWarperOcl>();
    }
#ifdef HAVE_OPENCV_CUDAWARPING
    else if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
    {
        if (warp_type == "plane")
            warper_creator = makePtr<cv::PlaneWarperGpu>();
        else if (warp_type == "cylindrical")
            warper_creator = makePtr<cv::CylindricalWarperGpu>();
        else if (warp_type == "spherical")
            warper_creator = makePtr<cv::SphericalWarperGpu>();
    }
    else
#endif
    {
        if (warp_type == "plane")
            warper_creator = makePtr<cv::PlaneWarper>();
        else if (warp_type == "cylindrical")
            warper_creator = makePtr<cv::CylindricalWarper>();
        else if (warp_type == "spherical")
            warper_creator = makePtr<cv::SphericalWarper>();
        else if (warp_type == "fisheye")
            warper_creator = makePtr<cv::FisheyeWarper>();
        else if (warp_type == "stereographic")
            warper_creator = makePtr<cv::StereographicWarper>();
        else if (warp_type == "compressedPlaneA2B1")
            warper_creator = makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
        else if (warp_type == "compressedPlaneA1.5B1")
            warper_creator = makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
        else if (warp_type == "compressedPlanePortraitA2B1")
            warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
        else if (warp_type == "compressedPlanePortraitA1.5B1")
            warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
        else if (warp_type == "paniniA2B1")
            warper_creator = makePtr<cv::PaniniWarper>(2.0f, 1.0f);
        else if (warp_type == "paniniA1.5B1")
            warper_creator = makePtr<cv::PaniniWarper>(1.5f, 1.0f);
        else if (warp_type == "paniniPortraitA2B1")
            warper_creator = makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
        else if (warp_type == "paniniPortraitA1.5B1")
            warper_creator = makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
        else if (warp_type == "mercator")
            warper_creator = makePtr<cv::MercatorWarper>();
        else if (warp_type == "transverseMercator")
            warper_creator = makePtr<cv::TransverseMercatorWarper>();
    }

    if (!warper_creator)
    {
        cout << "Can't create the following warper '" << warp_type << "'\n";
        return Mat();
    }

    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

    for (int i = 0; i < num_images; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
        K(0,0) *= swa; K(0,2) *= swa;
        K(1,1) *= swa; K(1,2) *= swa;

        corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);  //warp images
        //returned to corners[i] is a Point object, correspoding to projected image's top-left corner
        //warped image also saved in reference-passed images_warped[i]
        sizes[i] = images_warped[i].size();

        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);  //warp masks
    }

    vector<Mat> images_warped_f(num_images);
    for (int i = 0; i < num_images; ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

    LOGLN("Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    // Exposure Compensation, just a nicety , tries to allocate a lot of memory if too many input frames

    Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
    compensator->feed(corners, images_warped, masks_warped);  //memory error here if too many frames

    // Seam finding for compositing

    Ptr<SeamFinder> seam_finder;
    if (seam_find_type == "no")
        seam_finder = makePtr<detail::NoSeamFinder>();
    else if (seam_find_type == "voronoi")
        seam_finder = makePtr<detail::VoronoiSeamFinder>();
    else if (seam_find_type == "gc_color")
    {
#ifdef HAVE_OPENCV_CUDA
        if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
            seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR);
        else
#endif
            seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
    }
    else if (seam_find_type == "gc_colorgrad")
    {
#ifdef HAVE_OPENCV_CUDA
        if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
            seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
        else
#endif
            seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
    }
    else if (seam_find_type == "dp_color")
        seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR);
    else if (seam_find_type == "dp_colorgrad")
        seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR_GRAD);
    if (!seam_finder)
    {
        cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
        return Mat();
    }

    seam_finder->find(images_warped_f, corners, masks_warped);

    // Release unused memory
    images.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();


    LOGLN("Compositing...");
#if ENABLE_LOG
    t = getTickCount();
#endif

    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    Ptr<Blender> blender;
    //double compose_seam_aspect = 1;
    double compose_work_aspect = 1;

    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        //LOGLN("Compositing image #" << indices[img_idx]+1);

        // Read image and resize it if necessary
        //full_img = imread(img_names[img_idx]);
        //int frame_number = atoi((img_names[img_idx].substr(6)).c_str());
        full_img = frames[img_idx];  //original_frames[frame_number];
        //imshow("Display", full_img);
        //cout << "frame number " << frame_number << " " << img_names[img_idx] << endl;
        //waitKey(100);
        if (!is_compose_scale_set)
        {
            if (compose_megapix > 0)
                compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;

            // Compute relative scales
            //compose_seam_aspect = compose_scale / seam_scale;
            compose_work_aspect = compose_scale / work_scale;

            // Update warped image scale
            warped_image_scale *= static_cast<float>(compose_work_aspect);
            warper = warper_creator->create(warped_image_scale);

            // Update corners and sizes
            for (int i = 0; i < num_images; ++i)
            {
                // Update intrinsics
                cameras[i].focal *= compose_work_aspect;
                cameras[i].ppx *= compose_work_aspect;
                cameras[i].ppy *= compose_work_aspect;

                // Update corner and size
                Size sz = full_img_sizes[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                }

                Mat K;
                cameras[i].K().convertTo(K, CV_32F);
                Rect roi = warper->warpRoi(sz, K, cameras[i].R);
                corners[i] = roi.tl();
                sizes[i] = roi.size();
            }
        }
        if (abs(compose_scale - 1) > 1e-1)
            resize(full_img, img, Size(), compose_scale, compose_scale);
        else
            img = full_img;
        full_img.release();
        Size img_size = img.size();

        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);

        // Warp the current image
        warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

        // Warp the current image mask
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

        // Compensate exposure
        compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();

        dilate(masks_warped[img_idx], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size());
        mask_warped = seam_mask & mask_warped;

        if (!blender)  //Blending, again a nicety
        {
            blender = Blender::createDefault(blend_type, try_cuda);
            Size dst_sz = resultRoi(corners, sizes).size();
            float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
            if (blend_width < 1.f)
                blender = Blender::createDefault(Blender::NO, try_cuda);
            else if (blend_type == Blender::MULTI_BAND)
            {
                MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
                mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
                LOGLN("Multi-band blender, number of bands: " << mb->numBands());
            }
            else if (blend_type == Blender::FEATHER)
            {
                FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
                fb->setSharpness(1.f/blend_width);
                LOGLN("Feather blender, sharpness: " << fb->sharpness());
            }
            blender->prepare(corners, sizes);
        }

        // Blend the current image
        blender->feed(img_warped_s, mask_warped, corners[img_idx]);
    }  //End of compositing loop

    Mat result, result_mask;
    blender->blend(result, result_mask);

    LOGLN("Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    //imwrite(result_name + to_string(result_name_counter++) + ".jpg", result);

    //LOGLN("Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");

    //namedWindow("final");
    //imshow("final", result);
    //waitKey();

    //return 0;
    
    //Only needed if doing partitioned stitching
    //Mat converted_result;
    //result.convertTo(converted_result , CV_8UC3);
    //return converted_result;

    return result;
}
