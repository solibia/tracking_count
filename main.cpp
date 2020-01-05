/*#include "functions.h"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d.hpp"
//#include "opencv2/tracking.hpp"
#include <opencv2/video/tracking.hpp>

//#include "functions.h"

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iomanip>
#include <map>
#include <bits/basic_string.h>
*/

#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

//using namespace cv;



void OperationToDO(std::vector<std::string> detailCmdUser);

// methode principale du programme
int main(int argc, char** argv)
{
    std::string current_exec_name = argv[0]; // Name of the current exec program
       std::string first_arge;
       std::vector<std::string> all_args;
       if (argc > 1) {
         first_arge = argv[1];
         all_args.assign(argv + 1, argv + argc);//Regroupement des parametres de dans un vecteur de string
       }
     OperationToDO(all_args);

    return 0;
}
