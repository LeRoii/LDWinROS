#ifndef _HIKCAMERA_H_
#define _HIKCAMERA_H_

#include <opencv2/highgui/highgui.hpp>
#include "MvCameraControl.h"

class hikcamera
{
public:
    hikcamera();
    ~hikcamera();
    void imageCap(unsigned char* buf);

private:
    bool PrintDeviceInfo(MV_CC_DEVICE_INFO* pstMVDevInfo);
    
    void* m_handle;


};

#endif
