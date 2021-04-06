#차량지능기초 Report
=================

##### 1. 자율주행 인지에 관련된 3종 이상의 공개 Data Set 조사, 정리
##### 2. 자율주행 인지에 관련된 2종 이상 Open Source 조사, 정리
##### 3. #2의 정리한 코드 중 하나 실행해서 결과 확인
-----------------------------------------------------------------------



---:이름: 봉원준
---:학번: 20195320
---:학과: 자동차공학과
---:제출일: 2021-04-06









![image](https://user-images.githubusercontent.com/81850912/113655363-9c08fd00-96d4-11eb-9b3c-b9ecba6074d7.png)





##               ##자율주행 인지에 관련된 Data Set##
##  #1
<BBD100K: 대규모의 다양한 주행 비디오 데이터베이스>

 BBD100K는 가상 주행 장면에 대한 새롭고 다양한 대규모 데이터셋이다. 해당 연구진들은 주석 처리된 10만개 이상의 다양한 비디오 클립에 주행 장면의 가장 큰 데이터셋을 수집하고 주석을 달았다. 각 비디오의 길이는 약 40초, 720p 및 30fps이다. 동영상에는 rough한 주행 궤적을 보여주기 위해 휴대폰으로 기록된 GPS/IMU 정보도 함께 제공된다. 해당 비디오는 사진1과 같이 미국의 다양한 위치에서 수집되었고 다양한 기상 조건과 주간 및 야간을 포함한 다양한 시간대에 대한 정보도 다루었다. 

 ![image](https://user-images.githubusercontent.com/81850912/113657016-05d6d600-96d8-11eb-9015-61bd6cb268d9.png)
-사진1-


![image](https://user-images.githubusercontent.com/81850912/113657030-0f603e00-96d8-11eb-87c9-42e8913f5a75.png)
-사진2-  (BBD100K의 데이터 특징)

다음은 데이터의 활용 모습이다.
1)	주석
 
![image](https://user-images.githubusercontent.com/81850912/113657056-1b4c0000-96d8-11eb-9e92-7d80a570e585.png)
-사진3-

각 동영상에서 주석은 이미지 태깅, 도로 객체 경계 상자, 운전 가능 영역, 차선 표시 및 풀 프레임 인스턴스 분할과 같은 여러 수준에서 레이블이 지정된다. 이를 통해 데이터 및 개체 통계의 다양성을 이해하는데 도움이 될 수 있다.

2)	도로 물체 감지
 
![image](https://user-images.githubusercontent.com/81850912/113657165-50585280-96d8-11eb-9d6b-e6be7fe2c346.png)

-사진4-

객체의 분포와 위치를 이해하기 위해 100,000개의 키 프레임에서 도로에 일반적으로 나타나는 객체에 대해 객체 경계 상자에 레이블을 지정한다. 사진4의 막대 차트는 개체 수를 보여준다. 

3)	차선 표시
 
![image](https://user-images.githubusercontent.com/81850912/113657181-5a7a5100-96d8-11eb-88d1-1bec973a432d.png)

-사진5-

위치 정보에 대한 통신을 받지 못할 때 차선 표시는 자율 주행 시스템의 주행 방향 및 위치 결정에 중요한 단서로 작용한다. BBD100K에서 차선 표시 유형은 두 가지로 나뉜다. 수직 차선 표시 (사진5에서 빨간색 선으로 표시)는 차선의 주행 방향을 따르는 표시를 나타낸다. 평행 차선 표시 (사진5에서 파란색 선으로 표시)는 차선에 있는 차량이 정지해야 하는 것을 나타낸다.
4)	운전 가능 영역
 
![image](https://user-images.githubusercontent.com/81850912/113657214-682fd680-96d8-11eb-87e4-14f017946d78.png)

-사진6-

우리는 주행 시 차선 표시와 교통 장치에만 의존하지 않는다. 또한 도로를 공유하는 다른 물체와의 복잡한 상호작용에도 신경을 써야한다. 결국 어떤 영역으로 운행이 가능한지 이해하는 것이 중요하다. 이 문제를 해결하기 위해 사진6에서와 같이 주행 가능 영역의 세분화 주석도 제공한다. 본인 차량의 궤적에 따라 운전 가능 영역을 직접 운전 가능과 대체 운전 가능의 두 가지 범주로 나눈다. 빨간색으로 표시된 직접 운전 기능은 본인 차량이 도로 우선 순위를 가지며 해당 영역에서 계속 운전할 수 있음을 의미하고 파란색으로 표시된 영역은 본인 차량이 해당 영역에서 운전할 수 있지만 도로의 우선 순위가 다른 차량에 비해 낮으므로 주의해야 하는 영역이다.





## #2
<KITTI dataset>

![image](https://user-images.githubusercontent.com/81850912/113657869-d88b2780-96d9-11eb-8fe2-f788dea15181.png)
 -사진9-(Annieway)

KITTI dataset은 Annieway(사진9)를 이용해 운행을 하면서 거리의 영상을 깊이 값과 함께 담아낸 데이터셋이다.
KIT(karlsruhe institute of Technology)에서는 자율주행 플랫폼 Annieway를 활용하여 데이터를 추출한다. KIT에서는 스테레오, optical flow, 비주얼적 주행 거리 측정, 3D 물체 감지 및 3D 추적을 다룬다. 이를 위해 고해상도 컬러 및 그레이 스케일 비디오 카메라 2대가 장착된 표준 스테이션 왜건을 이용한다. Velodyne레이저 스케너와 GPS 위치 확인 시스템을 통해 정확한 GroundTruth가 제공된다. 데이터셋은 중형 도시인 Karlsruhe 주변을 운전하여 얻었다. 이미지 당 최대 15대의 자동차와 30명의 보행자가 보인다. 모든 데이터를 원시 형식으로 제공하는 것 외에도 각 작업에 대한 벤치 마크를 추출한다.
 
![image](https://user-images.githubusercontent.com/81850912/113657905-ef317e80-96d9-11eb-8559-983a4030d202.png)
-사진10-
 
![image](https://user-images.githubusercontent.com/81850912/113657928-f9537d00-96d9-11eb-8558-c9a7ec904698.png)
-사진11-
 
![image](https://user-images.githubusercontent.com/81850912/113657940-02444e80-96da-11eb-8b85-65b9be895dda.png)
-사진12-

![image](https://user-images.githubusercontent.com/81850912/113657957-0c664d00-96da-11eb-9112-ab1943012c9a.png) 
-사진13-
 
![image](https://user-images.githubusercontent.com/81850912/113657968-1720e200-96da-11eb-8517-3deb71fc931e.png)
-사진14-

데이터셋의 구성은 다음과 같다
-Stereo 2015/ flow 2015/ scene flow 2015 data set (2 GB)
-Annotated depth map data set (14 GB)
-Projected raw LiDaR scans data set (5 GB)
-Manually selected validation and test data sets (2 GB)
-Odometry data set (grayscale, 22 GB)
-Left color images of object data set (12 GB)
-Left color images of tracking data set (15 GB)





## #3
<nuScenes: 자율주행을 위한 멀티 모달 데이터셋>

nuTonomy scenes 또는 nuScenes는 자율주행을 위한 대규모 공개 데이터셋이다. 이 데이터셋은 실제 자율주행 자동차의 완전한 센서 세트를 장착해 실험적으로 도시 주행 상황을 연구할 수 있다. 360도 시야를 가진 카메라 6대, 레이더 5대, 라이다 1대 등 완전 자율형 차량 센서 제품군을 탑재한 최초의 데이터셋이다. nuScenes는 1000개의 장면으로 구성되며, 각 장면의 길이는 20초이고 23개 클래스와 8개 속성에 대해 3D 바운딩 박스로 주석을 달았다.
![image](https://user-images.githubusercontent.com/81850912/113658086-551e0600-96da-11eb-839a-f122ffddbb00.png)
 -사진7-

1) 데이터 수집
- 장면계획-
nuScenes 데이터 세트의 경우 보스턴과 싱가포르에서 약 15 시간의 운전 데이터를 수집한다. 전체 nuScenes 데이터 세트의 경우 Boston Seaport 및 싱가포르의 One North , Queenstown 및 Holland Village 지구에서 데이터를 게시한다 . 까다로운 시나리오를 포착하기 위해 운전 경로가 신중하게 선택된다. 다양한 위치, 시간 및 기상 조건을 목표로 한다. 클래스 빈도 분포의 균형을 맞추기 위해 희귀 클래스 (예 : 자전거)가 있는 장면을 더 많이 포함한다. 이러한 기준을 사용하여 각각 20 초 길이의 장면 1000 개를 수동으로 선택한다.

-자동차 설정 -
![image](https://user-images.githubusercontent.com/81850912/113658130-6a933000-96da-11eb-9f44-b5ffb35c8375.png)
-사진8-

보스턴과 싱가포르에서 운전하기 위해 센서 레이아웃이 동일한 두 대의 르노조이 자동차를 사용한다. 데이터는 연구 플랫폼에서 수집되었으며 Motional 제품에 사용된 설정을 나타내지 않는다. 센서 배치는 위의 사진8을 참조하십시오. 다음은 센서 데이터를 공개합니다.
 
 
2) 데이터 형식       
![image](https://user-images.githubusercontent.com/81850912/113658161-7b43a600-96da-11eb-9130-b4ebb1d36d6a.png)
 ![image](https://user-images.githubusercontent.com/81850912/113658207-94e4ed80-96da-11eb-8731-23292aed936d.png)
-사진9-
![image](https://user-images.githubusercontent.com/81850912/113658282-af1ecb80-96da-11eb-8dec-7581eb9b7494.png)

   3) 데이터 주석
주행 데이터를 수집한 후 2Hz에서 잘 동기화 된 키 프레임 (이미지, LIDAR, RADAR)을 샘플링 하여 주석 파트너 Scale 로 전송하여 주석을 처리한다. 전문 어노데이터와 여러 검증 단계를 사용하여 매우 정확한 어노테이션을 달성한다. nuScenes 데이터 세트의 모든 객체는 의미론적 범주와 함께 3D 경계 상자 및 이들이 발생하는 각 프레임에 대한 속성을 제공한다. 2D 경계 상자와 비교하여 공간에서 객체의 위치와 방향을 정확하게 추론할 수 있다.





-출처-

https://rdx-live.tistory.com/90 (자율주행 오픈소스 데이터셋)
https://bair.berkeley.edu/blog/2018/05/30/bdd/ (BBD100K)
https://www.nuscenes.org/nuscenes?tutorial=nuscenes (nuScenes)
http://www.cvlibs.net/datasets/kitti/ (KITTI)



