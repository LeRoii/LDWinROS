
#include <std_msgs/Empty.h>
#include <sensor_msgs/NavSatFix.h>
#include <log4cxx/logger.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>
#include <fstream>

#include "yolo_v2_class.hpp"


#include "ros/ros.h"
#include "hikcamera.h"



//回调函数
void callback(const sensor_msgs::NavSatFix& msg){
    //ROS_INFO("NearPoint: numaber:%d", msg.latitude);
     //ROS_INFO_STREAM("callback" << msg->data);
 }

 void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names,
    int current_det_fps = -1, int current_cap_fps = -1)
{
    int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };

    for (auto &i : result_vec) {
        cv::Scalar color = obj_id_to_color(i.obj_id);
        cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
        if (obj_names.size() > i.obj_id) {
            std::string obj_name = obj_names[i.obj_id];
            if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
            cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
            int max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
            max_width = std::max(max_width, (int)i.w + 2);
            //max_width = std::max(max_width, 283);
            std::string coords_3d;
            if (!std::isnan(i.z_3d)) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << "x:" << i.x_3d << "m y:" << i.y_3d << "m z:" << i.z_3d << "m ";
                coords_3d = ss.str();
                cv::Size const text_size_3d = getTextSize(ss.str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, 1, 0);
                int const max_width_3d = (text_size_3d.width > i.w + 2) ? text_size_3d.width : (i.w + 2);
                if (max_width_3d > max_width) max_width = max_width_3d;
            }

            cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 35, 0)),
                cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
                color, CV_FILLED, 8, 0);
            putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
            if(!coords_3d.empty()) putText(mat_img, coords_3d, cv::Point2f(i.x, i.y-1), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 0), 1);
        }
    }
    if (current_det_fps >= 0 && current_cap_fps >= 0) {
        std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) + "   FPS capture: " + std::to_string(current_cap_fps);
        putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
    }
}

void show_console_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names, int frame_id = -1) {
    if (frame_id >= 0) std::cout << " Frame: " << frame_id << std::endl;
    for (auto &i : result_vec) {
        if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
        std::cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y
            << ", w = " << i.w << ", h = " << i.h
            << std::setprecision(3) << ", prob = " << i.prob << std::endl;
    }
}

std::vector<std::string> objects_names_from_file(std::string const filename) {
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for(std::string line; getline(file, line);) file_lines.push_back(line);
    std::cout << "object names loaded \n";
    return file_lines;
}


int main (int argc, char** argv){
    ros::init(argc, argv, "camcap_node");
    ros::NodeHandle n;


    log4cxx::Logger::getLogger(ROSCONSOLE_DEFAULT_NAME)->setLevel(
    ros::console::g_level_lookup[ros::console::levels::Debug]);
    ros::console::notifyLoggerLevelsChanged();
    
    hikcamera camerainst;

    image_transport::ImageTransport it(n);
    image_transport::Publisher pub = it.advertise("camera/image", 1);

    cv::Mat image;
    unsigned char*  pFrameBuf = NULL;    

    pFrameBuf = (unsigned char*)malloc(1024*1280*3);  //rgb8
    //pFrameBuf = (unsigned char*)malloc(1024*1280*2);    //yuv422_yuyv



//yolo detect
    // std::string  names_file = "/home/xavier2/code/jetson_bot/src/cameracap/cfg/coco.names";
    // std::string  cfg_file = "/home/xavier2/code/jetson_bot/src/cameracap/cfg/yolov4.cfg";
    // std::string  weights_file = "/home/xavier2/code/jetson_bot/src/cameracap/cfg/yolov4.weights";

	// //std::string  names_file = "/home/xavier2/code/jetson_bot/src/cameracap/cfg/coco.names";
    // //std::string  cfg_file = "/home/xavier2/code/jetson_bot/src/cameracap/cfg/yolov4.cfg";
    // //std::string  weights_file = "/home/xavier2/code/jetson_bot/src/cameracap/cfg/yolov4.weights";
    
	// std::string filename;

    // Detector detector(cfg_file, weights_file);
    // auto obj_names = objects_names_from_file(names_file);

    //auto start = std::chrono::steady_clock::now();
    //std::vector<bbox_t> result_vec = detector.detect_resized(*det_image, mat_img.size().width, mat_img.size().height);
    //auto end = std::chrono::steady_clock::now();
    //std::chrono::duration<double> spent = end - start;
    //std::cout << " Time: " << spent.count() << " sec \n";

    //draw_boxes(mat_img, result_vec, obj_names);
//yolo detect end

    ros::Rate loop_rate(30);

	// FILE* fp = fopen("/home/xavier2/camraw.raw", "wb");
	// if (NULL == fp)
	// {
	// 	printf("fopen failed\n");
	// 	return 0;
	// }
	
	
    while(ros::ok()){
        //ROS_DEBUG_STREAM("cammm");
    
        camerainst.imageCap(pFrameBuf);

        /*
        cv::Mat matt = cv::Mat(1024,1280, CV_8UC3, pFrameBuf);
        cv::resize(matt,matt,cv::Size(640,512),(0,0),(0,0),cv::INTER_LINEAR);
        cvtColor(matt, matt, cv::COLOR_RGB2BGR);
        */
        
        //cv::Mat cameraRawImg = cv::Mat(1024,1280, CV_8UC2, pFrameBuf);
        //cvtColor(cameraYUVImg, cameraBGRImg, cv::COLOR_YUV2BGR_YUYV);
		
		cv::Mat cameraRawImg = cv::Mat(1024,1280, CV_8UC3, pFrameBuf);
		cvtColor(cameraRawImg, cameraRawImg, cv::COLOR_BGR2RGB);
        ROS_DEBUG_STREAM("imageCap ok");

		auto start = std::chrono::steady_clock::now();
        //auto det_image = detector.mat_to_image_resize(cameraRawImg);

        //std::vector<bbox_t> result_vec = detector.detect_resized(*det_image, cameraRawImg.size().width, cameraRawImg.size().height);
        //auto end = std::chrono::steady_clock::now();
        //std::chrono::duration<double> spent = end - start;
        //std::cout << " Time: " << spent.count() << " sec \n";

        //draw_boxes(cameraRawImg, result_vec, obj_names);
		
        //cv::imwrite("/home/xavier2/yoloout.png",cameraRawImg);
		cv::Mat yuvImg;
		cvtColor(cameraRawImg, yuvImg, cv::COLOR_RGB2YUV_I420);
		//fwrite(yuvImg.data, 1, 1024*1280*3/2, fp);


        //cv::imshow("111",cameraRawImg);
        //cv::waitKey(0);

        ROS_DEBUG_STREAM("imageCap ok");
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cameraRawImg).toImageMsg();
        pub.publish(msg);
        ros::spinOnce();
        loop_rate.sleep();

    }
// fclose(fp);
 }
