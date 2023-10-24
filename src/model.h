#pragma once
#include <opencv2/core/core.hpp>

namespace fight {

const int Size_ =  100;

typedef enum
{
    IA_TYPE_INTRUSION_DETECTION, //周边保护，入侵报警
    IA_TYPE_DIRECTION_DETECTION, //越线报警
    IA_TYPE_CLOSEDOOR_DETECTION, //
    IA_TYPE_LEAKWATER_DETECTION, //
    IA_TYPE_REMOVEOBJ_DETECTION, //
    IA_TYPE_PEOPLE_COUNT,		 //区域人员数量统计
    IA_TYPE_LOADSTONE_DETECTION, //遗留物报警
    IA_TYPE_PEOPLEDOWN_DETECTION,  //
    IA_TYPE_PEOPLETRACK_DETECTION, //
    IA_TYPE_PEOPLESLEEP_DETECTION, //
    IA_TYPE_PEOPLEPHONE_DETECTION, //人员打电话
    IA_TYPE_PEOPLEHELME_DETECTION,  //安全帽检测
    IA_TYPE_PEOPLESMOKE_DETECTION,
    IA_TYPE_VEHICLESTOP_DETECTION,
    IA_TYPE_FIREORSMOKE_DETECTION,
    IA_TYPE_LOOKWATER_DETECTION

} EIAType;

typedef struct
{
    int x;
    int y;

} cv_Point;

typedef struct
{
    cv_Point p[Size_]; //用于画出ROI的点，请顺时针给出，目前支持四边形
	std::vector<int> pointNum;		// 多边形点的个数
    int regionNum;  // 多边形个数

    float lowtime;
    float hightime;

    cv_Point c0[2];
    cv_Point c1[2];
    cv_Point c2[2];

    int alarm;		   // 0代表正常，1 代表异常
    int Frameinterval; //隔帧数量

    int FrameNum; //֡统计帧的个数
    int countNum; //֡人数统计阈值

    int behave; //算法功能使用关键部门，比如区域里面统计个数，识别算法，风井防爆盖的爆破。
    int cflag;	// 相机的景深标定的使用。备用

    float scaleX; //图像的X方向的缩放比例系数（0-1）之间的数值
    float scaleY; //图像的Y方向的缩放比例系数（0-1）之间的数值

    int width;
    int height;

    void *iModel;

} cvModel;


extern cvModel* Allocate_Algorithm(cv::Mat &input_frame, int algID, int gpuID);
extern void SetPara_Algorithm(cvModel *pModel,int algID);
extern void UpdateParams_Algorithm(cvModel *pModel);
extern void Process_Algorithm(cvModel *pModel, cv::Mat &input_frame);
extern void Destroy_Algorithm(cvModel *pModel);

} // namespace fight
