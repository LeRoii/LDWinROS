#include <stdio.h>
//#include <stdlib.h>
#include <string.h> //memset
#include <thread>
#include <unistd.h> //sleep
#include <opencv2/imgproc/imgproc.hpp>

#include "hikcamera.h"

using std::thread;

bool hikcamera::PrintDeviceInfo(MV_CC_DEVICE_INFO* pstMVDevInfo)
{
    if (NULL == pstMVDevInfo)
    {
        printf("The Pointer of pstMVDevInfo is NULL!\n");
        return false;
    }
    if (pstMVDevInfo->nTLayerType == MV_GIGE_DEVICE)
    {
        int nIp1 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24);
        int nIp2 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16);
        int nIp3 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8);
        int nIp4 = (pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff);

        // ch:打印当前相机ip和用户自定义名字 | en:print current ip and user defined name
        printf("Device Model Name: %s\n", pstMVDevInfo->SpecialInfo.stGigEInfo.chModelName);
        printf("CurrentIp: %d.%d.%d.%d\n" , nIp1, nIp2, nIp3, nIp4);
        printf("UserDefinedName: %s\n\n" , pstMVDevInfo->SpecialInfo.stGigEInfo.chUserDefinedName);
    }
    else if (pstMVDevInfo->nTLayerType == MV_USB_DEVICE)
    {
        printf("Device Model Name: %s\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.chModelName);
        printf("UserDefinedName: %s\n\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.chUserDefinedName);
    }
    else
    {
        printf("Not support.\n");
    }

    return true;
}

hikcamera::hikcamera()
{
    int nRet = MV_OK;

    m_handle = NULL;
    MV_CC_DEVICE_INFO_LIST stDeviceList;
    memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));

    // 枚举设备
    // enum device
    nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
    if (MV_OK != nRet)
    {
        printf("MV_CC_EnumDevices fail! nRet [%x]\n", nRet);
        return;
    }

    if (stDeviceList.nDeviceNum > 0)
    {
        for (int i = 0; i < stDeviceList.nDeviceNum; i++)
        {
            printf("[device %d]:\n", i);
            MV_CC_DEVICE_INFO* pDeviceInfo = stDeviceList.pDeviceInfo[i];
            if (NULL == pDeviceInfo)
            {
                return;
            } 
            PrintDeviceInfo(pDeviceInfo);            
        }  
    } 
    else
    {
        printf("Find No Devices!\n");
        return;
    }

    unsigned int nIndex = 0;

    // 选择设备并创建句柄
    // select device and create handle
    nRet = MV_CC_CreateHandle(&m_handle, stDeviceList.pDeviceInfo[nIndex]);
    if (MV_OK != nRet)
    {
        printf("MV_CC_CreateHandle fail! nRet [%x]\n", nRet);
        return;
    }

    // 打开设备
    // open device
    nRet = MV_CC_OpenDevice(m_handle);
    if (MV_OK != nRet)
    {
        printf("MV_CC_OpenDevice fail! nRet [%x]\n", nRet);
        return;
    }

    // 获取enum型变量
    // get IEnumeration variable
    MVCC_ENUMVALUE stEnumVal = {0};
    nRet = MV_CC_GetEnumValue(m_handle, "GainAuto", &stEnumVal);
    if (MV_OK == nRet)
    {
        printf("GainAuto current value:%d\n", stEnumVal.nCurValue);

        printf("supported GainAuto number:%d\n", stEnumVal.nSupportedNum);

    for (unsigned int i = 0; i < stEnumVal.nSupportedNum; ++i)
    {
        printf("supported GainAuto [%d]:%d\n", i, stEnumVal.nSupportValue[i]);
    }
        printf("\n");
    }
    else
    {
        printf("get GainAuto failed! nRet [%x]\n\n", nRet);
    }

    MVCC_INTVALUE stIntvalue = {0};
    nRet = MV_CC_GetIntValue(m_handle, "AutoExposureTimeupperLimit", &stIntvalue);
    if (MV_OK == nRet)
    {
        printf("AutoExposureTimeupperLimit:%d us!\n\n", stIntvalue.nCurValue);
    }

    // 开始取流
    // start grab image
    nRet = MV_CC_StartGrabbing(m_handle);
    if (MV_OK != nRet)
    {
        printf("MV_CC_StartGrabbing fail! nRet [%x]\n", nRet);
        return;
    }

    //cv::Mat mat;
    //thread thImageGrabber(&hikcamera::imageCap, this,std::ref(mat));
    //thImageGrabber.detach();

    
}

hikcamera::~hikcamera()
{
    int nRet = MV_OK;
    // 关闭设备
    // close device
    nRet = MV_CC_CloseDevice(m_handle);
    if (MV_OK != nRet)
    {
        printf("MV_CC_CloseDevice fail! nRet [%x]\n", nRet);
        return;
    }

    // 销毁句柄
    // destroy handle
    nRet = MV_CC_DestroyHandle(m_handle);
    if (MV_OK != nRet)
    {
        printf("MV_CC_DestroyHandle fail! nRet [%x]\n", nRet);
        return;
    }
}

void hikcamera::imageCap(unsigned char*  pFrameBuf)
{
    int nRet = MV_OK;
    MVCC_INTVALUE stIntvalue = {0};
    //获取一帧数据的大小
    nRet = MV_CC_GetIntValue(m_handle, "PayloadSize", &stIntvalue);
    if (nRet != MV_OK)
    {
        printf("Get PayloadSize failed! nRet [%x]\n", nRet);
        return;
    }
    int nBufSize = stIntvalue.nCurValue; //一帧数据大小
    int nRGBBufSize = 1024*1280*4+2048;

    unsigned int    nTestFrameSize = 0;
    //unsigned char*  pFrameBuf = NULL;
    
    MV_FRAME_OUT_INFO_EX stInfo;
    memset(&stInfo, 0, sizeof(MV_FRAME_OUT_INFO_EX));

    //上层应用程序需要根据帧率，控制好调用该接口的频率
    //此次代码仅供参考，实际应用建议另建线程进行图像帧采集和处理
    if (nTestFrameSize > 99) 
    {
        return;
    }

    //printf("nBufSize:%d\n", nBufSize);
    //nRet = MV_CC_GetImageForRGB(handle, pFrameBuf, nRGBBufSize, &stInfo, 1000);
    nRet = MV_CC_GetOneFrameTimeout(m_handle, pFrameBuf, nBufSize, &stInfo, 1000);

    if (MV_OK != nRet)
    {
        printf("MV_CC_GetImageForRGB not ok,%#x:\n", nRet); 
        sleep(10);
    }
    else
    {
        //...图像数据处理
        nTestFrameSize++;
        //printf("Now you GetOneFrame, Width[%d], Height[%d], nFrameNum[%d], pixelType:%#x\n\n", 
        //    stInfo.nWidth, stInfo.nHeight, stInfo.nFrameNum, stInfo.enPixelType); 

        // cv::Mat srclmage = cv::imread("../1.png"); 
        //cv::Mat matt = cv::Mat(stInfo.nHeight, stInfo.nWidth, CV_8UC3, pFrameBuf);
        //printf("Now ;;;;;;;");
        // cv::Mat dst;
        //cvtColor(matt, mat, cv::COLOR_RGB2BGR);
        //mat = dst.clone();
        //cv::imshow("111",matt);
        //cv::waitKey(0);
    }
}
