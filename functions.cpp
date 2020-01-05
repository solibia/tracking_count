//#include "functions.h"

/*
Functions()
{

}*/

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/video/tracking.hpp"

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

using namespace std;
using namespace cv;

typedef struct
{
    //Point 1 du rectangle entourant l'objet
    int x1;
    int y1;
    //Point 2 du rectangle entourant l'objet
    int x2;
    int y2;
    int indice;

} Shape;

//Functions() { }

//~Functions() { }

vector<int> getHeightWith(string fileName){
    // Chargement de la video
    VideoCapture videoCapture(fileName);
    vector<int> hw_vector;
    hw_vector.clear();
    if (!videoCapture.isOpened())
    {
        cout << "Impossible de lire la video : " << fileName << endl;
        exit(0);
    }else{
        // Dimensions des images de la video
        hw_vector.push_back(int(videoCapture.get(CV_CAP_PROP_FRAME_HEIGHT)));
        hw_vector.push_back(int(videoCapture.get(CV_CAP_PROP_FRAME_WIDTH)));
    }
    return  hw_vector;
}

//methode qui permet de determiner la boite englobante
vector<Shape> getShape(const Mat &cur_frame)
{
    // Liste des objets se trouvant dans le frame actuel
    vector<Shape> shapes_vector;
    shapes_vector.clear();
    vector<vector<Point> > contours;
    Mat frame = cur_frame.clone();
    // Déterminer les contours des différents objets du frame
    findContours(frame, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < (int) contours.size(); i++){
        if(contourArea(contours[i]) > 500) {
            Mat pointMat(contours[i]);
            //On considere chaque contour comme un objet
            Rect rect = boundingRect(pointMat);

            Shape shape;
            shape.x1 = rect.x;
            shape.y1 = rect.y;
            shape.x2 = rect.x + rect.width;
            shape.y2 = rect.y + rect.height;
            shape.indice = -1;
            shapes_vector.push_back(shape);
        }

    }
    return shapes_vector;
}

// methode permettant de dessiner la boite englobante
void drawShape(Mat &cur_frame, vector<Shape> shapes_vector)
{
    //Utiliser le vert pour dessiner des boites entourant des objets
    for (int i = 0; i < (int) shapes_vector.size(); i++)
    {
        rectangle(cur_frame,
                  Point(shapes_vector[i].x1,
                        shapes_vector[i].y1),
                  Point(shapes_vector[i].x2,
                        shapes_vector[i].y2), CV_RGB(0, 255, 0), 1);
        /*float my =(288/2) - (shapes_vector[i].y1+shapes_vector[i].y2)/2; //-384 × 288 ;
        if(my < 0){
            cout<<"Down \n";
        }else {
            cout<<"Up \n";
        }*/
    }
}

// methode permettant de dessiner la boite englobante
void countPeople(vector<Shape> shapes_vector, string video_path)
{
    vector<int> hw = getHeightWith(video_path);
    //Utiliser le vert pour dessiner des boites entourant des objets
    for (int i = 0; i < (int) shapes_vector.size(); i++)
    {
        /*rectangle(cur_frame,
                  Point(shapes_vector[i].x1,
                        shapes_vector[i].y1),
                  Point(shapes_vector[i].x2,
                        shapes_vector[i].y2), CV_RGB(0, 255, 0), 1);*/
        float my =(hw.at(0)/2)-shapes_vector[i].y2; //-384 × 288 ;
        //cout<<my;
        //cout<<"\n";
        //cout<<hw.at(0);
        //cout<<hw.at(1);
        if(my == 0){
            //cout<<shapes_vector[i].y1;
            //cout<<"\n";
            //cout<<shapes_vector[i].y2;
            //cout<<"\n";
            cout<<my;
            cout<<"On line \n";
        }else if(my < 0 && my > -1){
            //cout<<shapes_vector[i].y1;
            //cout<<"\n";
            //cout<<shapes_vector[i].y2;
            //cout<<"\n";
            cout<<my;
            cout<<"Down \n";
        }else if(my < 1 && my > 0) {
            //cout<<shapes_vector[i].y1;
            //cout<<"\n";
            //cout<<shapes_vector[i].y2;
            //cout<<"\n";
            cout<<my;
            cout<<"Up \n";
        }
    }
}



// Fonction pour dessiner des cercles sur l'image
void drawCircle(Mat &fol_pic, Point centre, int d, const Scalar& color)
{
    circle(fol_pic, centre, d, color, 1, 0);
}

// Fonction pour dessiner des carrés sur l'image
void drawSquare(Mat &fol_pic, Point centre, int d, const Scalar& color)
{
    rectangle(fol_pic, Point(centre.x - d, centre.y - d),
              Point(centre.x + d, centre.y + d), color, 1, 0);
}

// Fonction pour dessiner des croix sur l'image
void drawCross(Mat &fol_pic, Point centre, int d, const Scalar& color)
{
    //La forme est x pour chaque point
    line(fol_pic, Point(centre.x - d, centre.y - d),
         Point(centre.x + d, centre.y + d), color, 1, 0);
    line(fol_pic, Point(centre.x + d, centre.y - d),
         Point(centre.x - d, centre.y + d), color, 1, 0);
}

// Initialiser le Filtre de kalman
void kalmanFilterInitializer(map<int, KalmanFilter> &liste_filtres_kalman,
                                        vector<Shape> &shape_vector, int width,
                                        int height, string nom_video)
{
    int maxIndex = -1;

    //Determiner l'index maximal de l'objet dans la liste
    for (int i = 0; i < (int) shape_vector.size(); i++)
    {
        if (maxIndex < shape_vector[i].indice
                && shape_vector[i].indice != -1)
        {
            maxIndex = shape_vector[i].indice;
        }
    }

    Mat mesure = Mat::zeros(4, 1, CV_32FC1);

    //Declarer une image pour contenir tous les traces de tous les objets en mouvement
    stringstream ss;
    ss << "images_suivi/" << nom_video << ".png";
    string fileName = ss.str();
    Mat imgSuiviMouvement = imread(fileName, -1);

    for (int i = 0; i < (int) shape_vector.size(); i++)
    {

        // Initialiser le filter Kalman
        KalmanFilter filtre_kalman(4, 4, 0);

        // Initialiser des matrices
        setIdentity(filtre_kalman.transitionMatrix, cvRealScalar(1));
        filtre_kalman.transitionMatrix.at<float>(0, 2) = 1;
        filtre_kalman.transitionMatrix.at<float>(1, 3) = 1;
        setIdentity(filtre_kalman.processNoiseCov, cvRealScalar(0));
        setIdentity(filtre_kalman.measurementNoiseCov, cvRealScalar(0));
        setIdentity(filtre_kalman.measurementMatrix, cvRealScalar(1));
        setIdentity(filtre_kalman.errorCovPost, cvRealScalar(1));

        // Faire la prediction sans se baser sur les donnees historiques
        Mat predictionMat = filtre_kalman.predict();

        // Mesure
        mesure.at<float>(0, 0) = (shape_vector[i].x2
                                  + shape_vector[i].x1) / 2;
        mesure.at<float>(1, 0) = (shape_vector[i].y2
                                  + shape_vector[i].y1) / 2;
        // La vitesse vx, vy
        mesure.at<float>(2, 0) = 0;
        mesure.at<float>(3, 0) = 0;

        //Correction des mesures
        Mat correctionMat = filtre_kalman.correct(mesure);

        //Pour chaque objet, initialiser une image suivi
        Mat imgSuivi = Mat::zeros(height, width, CV_8UC3);

        if (shape_vector[i].indice == -1)
        {
            // Initialiser l'indice de l'objet
            maxIndex++;
            shape_vector[i].indice = maxIndex;

            //Dessiner le trajectoire de prediction
            drawCross(imgSuivi,
                                 Point(predictionMat.at<float>(0, 0),
                                       predictionMat.at<float>(1, 0)), 3,
                                 CV_RGB(0, 0, 255));
            drawCross(imgSuiviMouvement,
                                 Point(predictionMat.at<float>(0, 0),
                                       predictionMat.at<float>(1, 0)), 3,
                                 CV_RGB(0, 0, 255));

            //Dessiner le trajectoire de mesure
            drawCross(imgSuivi,
                                 Point(mesure.at<float>(0, 0), mesure.at<float>(1, 0)), 3,
                                 CV_RGB(0, 255, 0));
            drawCross(imgSuiviMouvement,
                                 Point(mesure.at<float>(0, 0), mesure.at<float>(1, 0)), 3,
                                 CV_RGB(0, 255, 0));

            //Dessiner le trajectoire de correction
            drawSquare(imgSuivi,
                                  Point(correctionMat.at<float>(0, 0),
                                        correctionMat.at<float>(1, 0)), 3,
                                  CV_RGB(255, 0, 0));
            drawSquare(imgSuiviMouvement,
                                  Point(correctionMat.at<float>(0, 0),
                                        correctionMat.at<float>(1, 0)), 3,
                                  CV_RGB(255, 0, 0));

            //Dessiner des textes dans l'image de suivi de mouvement
            stringstream ssText;
            ssText << shape_vector[i].indice;
            string text = ssText.str();
            int fontFace = CV_FONT_HERSHEY_SIMPLEX;
            double fontScale = 0.5;
            int thickness = 1;

            Point textPosition(mesure.at<float>(0, 0), mesure.at<float>(1, 0));
            putText(imgSuivi, text, textPosition, fontFace, fontScale,
                    CV_RGB(255, 255, 255), thickness, 8);

            //Enregistrer l'image suivi pour cet objet
            stringstream ss;
            ss << "images_suivi/" << nom_video << "_objet_"
               << shape_vector[i].indice << ".png";
            string filename = ss.str();
            //cout<<"Before";
            //imshow(fileName, imgSuivi);
            //cout<<"After";
            imwrite(filename, imgSuivi);

            //Ajouter le filtre kalman actuel à la liste des filtres
            liste_filtres_kalman[shape_vector[i].indice] =
                filtre_kalman;
        }

    }
    imwrite(fileName, imgSuiviMouvement);
}

// Fonction pour comparer les objets d'un frame précédent et du frame courant
int getObjectIndex(Shape objet_precedent,
                              vector<Shape> &vecteur_objets_actuels,
                              int seuil_correspondance)
{
    // Initialiser l'indice de l'objet à -1: c'est à dire ne correspondant à aucun objet
    int index = -1;

    float minDist = 1000000000.0;
    Shape objet_considere;
    int centreX1 = (objet_precedent.x1 + objet_precedent.x2) / 2;
    int centreY1 = (objet_precedent.y1 + objet_precedent.y2) / 2;

    for (int i = 0; i < (int) vecteur_objets_actuels.size(); i++)
    {
        int centreX = (vecteur_objets_actuels[i].x1
                       + vecteur_objets_actuels[i].x2) / 2;
        int centreY = (vecteur_objets_actuels[i].y1
                       + vecteur_objets_actuels[i].y2) / 2;

        // verifier la distance entre les objets: objet précedent et chaque objet de la liste
        float distance = sqrt( (centreX - centreX1) * (centreX - centreX1)
                             + (centreY - centreY1) * (centreY - centreY1));
        if (distance < minDist && distance < (float) seuil_correspondance)
        {
            minDist = distance;
            index = i;
            objet_considere = vecteur_objets_actuels[i];
        }
        // En cas d'egalité verifier la surface la plus petite

    }
    return index;
}

// methode d'amelioration de la detection de mouvement
Mat detectionImproved(Mat &motion_pic)
{
    // Erosion + dilatation
    Mat erode_elem = getStructuringElement(MORPH_RECT, Size(3, 3),
                                           Point(1, 1));
    Mat dilate_elem = getStructuringElement(MORPH_RECT, Size(3, 3),
                                            Point(1, 1));

    erode(motion_pic, motion_pic, erode_elem);
    dilate(motion_pic, motion_pic, dilate_elem);

    return motion_pic;
}


// methode de suivi de mouvement
void motionFollowing(string nom_video, Mat image_arriere_plan,
                                int seuil_detection1, int seuil_correspondance)
{

    vector<Mat> images_video;
    vector<Mat> images_mouvement;
    stringstream chemin_video;
    Mat imageSuivi;
    //imageMouv  = Mat::zeros(frame.size(), CV_8UC1);

    chemin_video << "videos/" << nom_video;
    string fileName = chemin_video.str();
    char key;

    // Vecteur contenant les objets détectés du frame precedent
    vector<Shape> ListObjetsPrecedents;
    ListObjetsPrecedents.clear();

    // Vecteur contenant les objets détectés dans le frame courant
    vector<Shape> ListObjetsActuel;
    ListObjetsActuel.clear();

    //Vecteur contenant tous les objets
    vector<Shape> ListObjetsTotal;
    ListObjetsTotal.clear();

    //List des Kalmam filter des objets en mouvement dans le frame courant
    map<int, KalmanFilter> listKalmanFilter;
    listKalmanFilter.clear();

    namedWindow(nom_video, CV_WINDOW_AUTOSIZE);
    namedWindow("Image Arriere Plan", CV_WINDOW_AUTOSIZE);
    namedWindow("Detection Mouvement", CV_WINDOW_AUTOSIZE);

    GaussianBlur(image_arriere_plan, image_arriere_plan, Size(5, 5), 0, 0);
    // Chargement de la video
    VideoCapture videoCapture(fileName);

    if (!videoCapture.isOpened())
    {
        cout << "Impossible de lire la video : " << nom_video << endl;
        exit(0);
    }
    else
    {
        int hauteur = videoCapture.get(CV_CAP_PROP_FRAME_HEIGHT);
        int largeur = videoCapture.get(CV_CAP_PROP_FRAME_WIDTH);

        // lecture des images constituant la video
        int numFrameActuel = 0;
        while (key != 'q' && key != 'Q')
        {
            Mat frame, frame_gray;
            videoCapture >> frame;
            if (!frame.data)
            {
                cout << "Fin de lecture de la video" << endl;
                break;
            }
            else
            {
                // Conversion des images en niveau de gris
                cvtColor(frame, frame_gray, CV_BGR2GRAY);
                // Enregistrement de l'image dans le vecteur d'image
                images_video.push_back(frame_gray);
                GaussianBlur(images_video.back(), images_video.back(),
                             Size(5, 5), 0, 0);
                Mat difference_images;
                // Différence entre image arriere-plan et image capturée pour détecter le mouvement
                absdiff(image_arriere_plan, images_video.back(),
                        difference_images);
                images_mouvement.push_back(difference_images);
                // Seuillage binaire pour éliminer les bruits
                threshold(images_mouvement.back(), images_mouvement.back(),
                          seuil_detection1, 255.0, CV_THRESH_BINARY);
                images_mouvement.back() = detectionImproved(images_mouvement.back());

                //détermination des boites englobantes pour les objets détectés
                vector<Shape> shape_vector = getShape(images_mouvement.back());
                drawShape(frame, shape_vector);


                if (images_mouvement.back().data)
                {
                    // Enregistrement de l'image de mouvement
                    stringstream path;
                    path << "images_mouvement/" << nom_video << "seuil_"
                         << seuil_detection1 << "_frame_"
                         << images_video.size() - 1 << ".png";
                    string fileName = path.str();

                    //Determiner des objets se trouvant dans le frame courant
                    ListObjetsActuel = getShape(images_mouvement.back());

                    //Dessiner des boites sur l'image correspondante
                    drawShape(frame, ListObjetsActuel);

                    imageSuivi = Mat::zeros(frame.size(), CV_8UC3);

                    //*****************************************  Suivi de mouvement   *************************************/
                    if (numFrameActuel == 0 && (ListObjetsActuel.size() > 0))
                    {
                        kalmanFilterInitializer(listKalmanFilter,
                                                           ListObjetsActuel, frame.cols, frame.rows,
                                                           nom_video);
                        ListObjetsPrecedents = ListObjetsActuel;
                        ListObjetsTotal = ListObjetsActuel;
                    }
                    else

                    {
                        stringstream ss;
                        ss << "images_suivi/" << nom_video << ".png";
                        string fileName = ss.str();

                        //cout<<"Before lu ";
                        imageSuivi = imread(fileName, -1); // chargement de l'image pour tracer le parcours
                        //cout<<"After lu";
                        //imshow(fileName,imageSuivi);

                        ListObjetsTotal.clear();

                        for (int i = 0; i < (int) ListObjetsPrecedents.size(); i++)
                        {
                            // 1ere Etape: Faire la prediction des positions des objets
                            Mat predictionMat =
                                listKalmanFilter[ListObjetsPrecedents[i].indice].predict();

                            stringstream ss;
                            ss << "images_suivi/" << nom_video << "_objet_"
                               << ListObjetsPrecedents[i].indice << ".png";
                            string fileName = ss.str();
                            Mat imgSuiviObj = imread(fileName, -1);

                            // Dessiner le trajectoire de prediction
                            drawCross(imgSuiviObj,
                                                 Point(predictionMat.at<float>(0, 0),
                                                       predictionMat.at<float>(1, 0)), 3,
                                                 CV_RGB(255, 0, 0));
                            drawCross(imageSuivi,
                                                  Point(predictionMat.at<float>(0, 0),
                                                        predictionMat.at<float>(1, 0)), 3,
                                                  CV_RGB(255, 0, 0));

                                                 // Chercher la correspondance entre les objets du frame précédent et ceux du frame actuel
                                                 int correspondance = getObjectIndex(
                                                             ListObjetsPrecedents[i], ListObjetsActuel,
                                                             seuil_correspondance);

                                                 if (correspondance != -1)
                        {
                            ListObjetsActuel[correspondance].indice =
                                    ListObjetsPrecedents[i].indice;

                                // 2e Etape: Mesure des positions des objets
                                Mat mesure = Mat::zeros(4, 1, CV_32FC1);

                                mesure.at<float>(0, 0) =
                                    (ListObjetsActuel[correspondance].x1
                                     + ListObjetsActuel[correspondance].x2)
                                    / 2;
                                mesure.at<float>(1, 0) =
                                    (ListObjetsActuel[correspondance].y1
                                     + ListObjetsActuel[correspondance].y2)
                                    / 2;

                                float vx = 0;
                                float vy = 0;

                                // Les coordonnées de la vitesse
                                vx = ((ListObjetsActuel[correspondance].x1
                                       + ListObjetsActuel[correspondance].x2) / 2)
                                     - ((ListObjetsPrecedents[i].x1
                                         + ListObjetsPrecedents[i].x2) / 2);

                                vy = ((ListObjetsActuel[correspondance].y1
                                       + ListObjetsActuel[correspondance].y2) / 2)
                                     - ((ListObjetsPrecedents[i].y1
                                         + ListObjetsPrecedents[i].y2) / 2);

                                mesure.at<float>(2, 0) = vx;
                                mesure.at<float>(3, 0) = vy;

                                // Afficher et enregistrer les resultats suivi du mouvement et dessiner le trajectoire de mesure
                                drawCircle(imageSuivi,
                                                      Point(mesure.at<float>(0, 0),
                                                            mesure.at<float>(1, 0)), 3,
                                                      CV_RGB(0, 255, 0));
                                drawCircle(imgSuiviObj,
                                                      Point(mesure.at<float>(0, 0),
                                                            mesure.at<float>(1, 0)), 3,
                                                      CV_RGB(0, 255, 0));

                                // 3e Etape: Faire la correction des positions des objets
                                Mat correctionMat =
                                    listKalmanFilter[ListObjetsPrecedents[i].indice].correct(
                                        mesure);

                                // Dessiner la trajectoire de la correction
                                drawSquare(imageSuivi,
                                                      Point(correctionMat.at<float>(0, 0),
                                                            correctionMat.at<float>(1, 0)), 3,
                                                      CV_RGB(0, 0, 255));
                                drawSquare(imgSuiviObj,
                                                      Point(correctionMat.at<float>(0, 0),
                                                            correctionMat.at<float>(1, 0)), 3,
                                                      CV_RGB(0, 0, 255));

                                //Enregistrer l'image suivi pour cet objet
                                imwrite(fileName, imgSuiviObj);
                                ListObjetsPrecedents[i] =
                                    ListObjetsActuel[correspondance];
                            }
                        }

                        ListObjetsTotal = ListObjetsPrecedents;

                        for (int i = 0; i < (int) ListObjetsActuel.size(); i++)
                        {
                            if (ListObjetsActuel[i].indice == -1)
                            {
                                int correspondance = getObjectIndex(
                                                         ListObjetsActuel[i], ListObjetsPrecedents,
                                                         seuil_correspondance);

                                if (correspondance != -1)
                                {
                                    ListObjetsActuel[i].indice =
                                        ListObjetsPrecedents[correspondance].indice;
                                }
                            }
                        }

                        for (int i = 0; i < (int) ListObjetsActuel.size(); i++)
                        {
                            if (ListObjetsActuel[i].indice == -1)
                                ListObjetsTotal.push_back(ListObjetsActuel[i]);
                        }

                        kalmanFilterInitializer(listKalmanFilter, ListObjetsTotal,
                                                  frame.cols, frame.rows, nom_video);
                        ListObjetsPrecedents = ListObjetsTotal;
                    }
                    //frame.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

                    //int thickness = 2;
                    //int lineType = LINE_8;
                    Point start = Point(0,int(hauteur/2));
                    Point end = Point(largeur,int(hauteur/2));

                    line( frame, start, end, Scalar( 0, 0, 0 ), 2, LINE_8 );

                    //Ajouter les noms des objets en mouvement dans le frame courant
                    for (int i = 0; i < (int) ListObjetsActuel.size(); i++)
                    {
                        stringstream ssTmp;
                        ssTmp << ListObjetsActuel[i].indice;
                        string text = ssTmp.str();

                        int fontFace = CV_FONT_HERSHEY_SIMPLEX;
                        double fontScale = 0.5;
                        int thickness = 1;

                        int x = (ListObjetsActuel[i].x1 + ListObjetsActuel[i].x2)
                                / 2;
                        int y = (ListObjetsActuel[i].y1 + ListObjetsActuel[i].y2)
                                / 2;
                        Point textPosition(x, y);
                        putText(frame, text, textPosition, fontFace, fontScale,
                                CV_RGB(0, 0, 255), thickness, 8);
                    }

                    //cout<<"Affichage et enregistrement des resultats images_suivi";
                    stringstream ss1;
                    ss1 << "images_suivi/" << nom_video << ".png";
                    string fileName1 = ss1.str();
                    if (!imwrite(fileName1, imageSuivi))
                        cout << "Erreur lors de l'enregistrement de line 624 " << fileName1
                             << endl;

                    stringstream ss2;
                    ss2 << "images_videos/" << nom_video << "_" << numFrameActuel
                        << ".png";
                    string fileName2 = ss2.str();
                    if (!imwrite(fileName2, frame))
                        cout << "Erreur lors de l'enregistrement de " << fileName2
                             << endl;

                    //enrégistrement de l'image de la vidéo avec les boites englobantes
                    stringstream ss3;
                    ss3 << "images_mouvement/" << nom_video << "seuil_" << seuil_detection1
                        << "_frame_" << images_video.size() - 1 << ".png";
                    string fileName3 = ss3.str();
                    if (!imwrite(fileName3, images_mouvement.back()))
                        cout << "Erreur lors de l'enregistrement de " << fileName3
                             << endl;

                    imshow("Suivi Mouvement", imageSuivi);

                    numFrameActuel++;

                    imshow(nom_video,frame);
                    imshow("Image Arriere Plan", image_arriere_plan);
                    imshow("Detection Mouvement", images_mouvement.back());

                    key = cvWaitKey(40);
                }
            }
        }
        cvDestroyAllWindows();
    }
}


// methode de comptage
void motionCount(string nom_video, Mat image_arriere_plan,
                                int seuil_detection1, int seuil_correspondance)
{

    vector<Mat> images_video;
    vector<Mat> images_mouvement;
    stringstream chemin_video;
    Mat imageSuivi;
    //imageMouv  = Mat::zeros(frame.size(), CV_8UC1);

    chemin_video << "videos/" << nom_video;
    string fileName = chemin_video.str();
    char key;

    // Vecteur contenant les objets détectés du frame precedent
    vector<Shape> ListObjetsPrecedents;
    ListObjetsPrecedents.clear();

    // Vecteur contenant les objets détectés dans le frame courant
    vector<Shape> ListObjetsActuel;
    ListObjetsActuel.clear();

    //Vecteur contenant tous les objets
    vector<Shape> ListObjetsTotal;
    ListObjetsTotal.clear();

    //List des Kalmam filter des objets en mouvement dans le frame courant
    map<int, KalmanFilter> listKalmanFilter;
    listKalmanFilter.clear();

    namedWindow(nom_video, CV_WINDOW_AUTOSIZE);
    namedWindow("Image Arriere Plan", CV_WINDOW_AUTOSIZE);
    namedWindow("Detection Mouvement", CV_WINDOW_AUTOSIZE);

    GaussianBlur(image_arriere_plan, image_arriere_plan, Size(5, 5), 0, 0);
    // Chargement de la video
    VideoCapture videoCapture(fileName);

    if (!videoCapture.isOpened())
    {
        cout << "Impossible de lire la video : " << nom_video << endl;
        exit(0);
    }
    else
    {
        int hauteur = videoCapture.get(CV_CAP_PROP_FRAME_HEIGHT);
        int largeur = videoCapture.get(CV_CAP_PROP_FRAME_WIDTH);

        // lecture des images constituant la video
        int numFrameActuel = 0;
        int totalPersonnes = 0;
        while (key != 'q' && key != 'Q')
        {
            Mat frame, frame_gray;
            videoCapture >> frame;
            if (!frame.data)
            {
                cout << "Fin de lecture de la video" << endl;
                break;
            }
            else
            {
                // Conversion des images en niveau de gris
                cvtColor(frame, frame_gray, CV_BGR2GRAY);
                // Enregistrement de l'image dans le vecteur d'image
                images_video.push_back(frame_gray);
                GaussianBlur(images_video.back(), images_video.back(),
                             Size(5, 5), 0, 0);
                Mat difference_images;
                // Différence entre image arriere-plan et image capturée pour détecter le mouvement
                absdiff(image_arriere_plan, images_video.back(),
                        difference_images);
                images_mouvement.push_back(difference_images);
                // Seuillage binaire pour éliminer les bruits
                threshold(images_mouvement.back(), images_mouvement.back(),
                          seuil_detection1, 255.0, CV_THRESH_BINARY);
                images_mouvement.back() = detectionImproved(images_mouvement.back());

                //détermination des boites englobantes pour les objets détectés
                vector<Shape> shape_vector = getShape(images_mouvement.back());
                drawShape(frame, shape_vector);

                if (images_mouvement.back().data)
                {
                    // Enregistrement de l'image de mouvement
                    stringstream path;
                    path << "images_mouvement/" << nom_video << "seuil_"
                         << seuil_detection1 << "_frame_"
                         << images_video.size() - 1 << ".png";
                    string fileName = path.str();

                    //Determiner des objets se trouvant dans le frame courant
                    ListObjetsActuel = getShape(images_mouvement.back());

                    //Dessiner des boites sur l'image correspondante
                    drawShape(frame, ListObjetsActuel);

                    imageSuivi = Mat::zeros(frame.size(), CV_8UC3);

                    //*****************************************  Suivi de mouvement   *************************************/
                    if (numFrameActuel == 0 && (ListObjetsActuel.size() > 0))
                    {
                        kalmanFilterInitializer(listKalmanFilter,
                                                           ListObjetsActuel, frame.cols, frame.rows,
                                                           nom_video);
                        ListObjetsPrecedents = ListObjetsActuel;
                        ListObjetsTotal = ListObjetsActuel;
                    }
                    else

                    {
                        stringstream ss;
                        ss << "images_suivi/" << nom_video << ".png";
                        string fileName = ss.str();

                        //cout<<"Before lu ";
                        imageSuivi = imread(fileName, -1); // chargement de l'image pour tracer le parcours
                        //cout<<"After lu";
                        //imshow(fileName,imageSuivi);

                        ListObjetsTotal.clear();

                        for (int i = 0; i < (int) ListObjetsPrecedents.size(); i++)
                        {
                            // 1ere Etape: Faire la prediction des positions des objets
                            Mat predictionMat =
                                listKalmanFilter[ListObjetsPrecedents[i].indice].predict();

                            stringstream ss;
                            ss << "images_suivi/" << nom_video << "_objet_"
                               << ListObjetsPrecedents[i].indice << ".png";
                            string fileName = ss.str();
                            Mat imgSuiviObj = imread(fileName, -1);

                            // Dessiner le trajectoire de prediction
                            drawCross(imgSuiviObj,
                                                 Point(predictionMat.at<float>(0, 0),
                                                       predictionMat.at<float>(1, 0)), 3,
                                                 CV_RGB(255, 0, 0));
                            drawCross(imageSuivi,
                                                  Point(predictionMat.at<float>(0, 0),
                                                        predictionMat.at<float>(1, 0)), 3,
                                                  CV_RGB(255, 0, 0));

                                                 // Chercher la correspondance entre les objets du frame précédent et ceux du frame actuel
                                                 int correspondance = getObjectIndex(
                                                             ListObjetsPrecedents[i], ListObjetsActuel,
                                                             seuil_correspondance);

                                                 if (correspondance != -1)
                        {
                            ListObjetsActuel[correspondance].indice =
                                    ListObjetsPrecedents[i].indice;

                                // 2e Etape: Mesure des positions des objets
                                Mat mesure = Mat::zeros(4, 1, CV_32FC1);

                                mesure.at<float>(0, 0) =
                                    (ListObjetsActuel[correspondance].x1
                                     + ListObjetsActuel[correspondance].x2)
                                    / 2;
                                mesure.at<float>(1, 0) =
                                    (ListObjetsActuel[correspondance].y1
                                     + ListObjetsActuel[correspondance].y2)
                                    / 2;

                                float vx = 0;
                                float vy = 0;

                                // Les coordonnées de la vitesse
                                vx = ((ListObjetsActuel[correspondance].x1
                                       + ListObjetsActuel[correspondance].x2) / 2)
                                     - ((ListObjetsPrecedents[i].x1
                                         + ListObjetsPrecedents[i].x2) / 2);

                                vy = ((ListObjetsActuel[correspondance].y1
                                       + ListObjetsActuel[correspondance].y2) / 2)
                                     - ((ListObjetsPrecedents[i].y1
                                         + ListObjetsPrecedents[i].y2) / 2);

                                mesure.at<float>(2, 0) = vx;
                                mesure.at<float>(3, 0) = vy;

                                // Afficher et enregistrer les resultats suivi du mouvement et dessiner le trajectoire de mesure
                                drawCircle(imageSuivi,
                                                      Point(mesure.at<float>(0, 0),
                                                            mesure.at<float>(1, 0)), 3,
                                                      CV_RGB(0, 255, 0));
                                drawCircle(imgSuiviObj,
                                                      Point(mesure.at<float>(0, 0),
                                                            mesure.at<float>(1, 0)), 3,
                                                      CV_RGB(0, 255, 0));

                                // 3e Etape: Faire la correction des positions des objets
                                Mat correctionMat =
                                    listKalmanFilter[ListObjetsPrecedents[i].indice].correct(
                                        mesure);

                                // Dessiner la trajectoire de la correction
                                drawSquare(imageSuivi,
                                                      Point(correctionMat.at<float>(0, 0),
                                                            correctionMat.at<float>(1, 0)), 3,
                                                      CV_RGB(0, 0, 255));
                                drawSquare(imgSuiviObj,
                                                      Point(correctionMat.at<float>(0, 0),
                                                            correctionMat.at<float>(1, 0)), 3,
                                                      CV_RGB(0, 0, 255));

                                //Enregistrer l'image suivi pour cet objet
                                imwrite(fileName, imgSuiviObj);
                                ListObjetsPrecedents[i] =
                                    ListObjetsActuel[correspondance];
                            }
                        }

                        ListObjetsTotal = ListObjetsPrecedents;

                        for (int i = 0; i < (int) ListObjetsActuel.size(); i++)
                        {
                            if (ListObjetsActuel[i].indice == -1)
                            {
                                int correspondance = getObjectIndex(
                                                         ListObjetsActuel[i], ListObjetsPrecedents,
                                                         seuil_correspondance);

                                if (correspondance != -1)
                                {
                                    ListObjetsActuel[i].indice =
                                        ListObjetsPrecedents[correspondance].indice;
                                }
                            }
                        }

                        for (int i = 0; i < (int) ListObjetsActuel.size(); i++)
                        {
                            if (ListObjetsActuel[i].indice == -1)
                                ListObjetsTotal.push_back(ListObjetsActuel[i]);
                        }

                        kalmanFilterInitializer(listKalmanFilter, ListObjetsTotal,
                                                  frame.cols, frame.rows, nom_video);
                        ListObjetsPrecedents = ListObjetsTotal;
                    }
                    //frame.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

                    //int thickness = 2;
                    //int lineType = LINE_8;
                    Point start = Point(0,int(hauteur/2));
                    Point end = Point(largeur,int(hauteur/2));

                    line( frame, start, end, Scalar( 0, 0, 0 ), 2, LINE_8 );

                    //Ajouter les noms des objets en mouvement dans le frame courant
                    string totalCount;
                    for (int i = 0; i < (int) ListObjetsActuel.size(); i++)
                    {
                        float my =(hauteur/2)-ListObjetsActuel[i].y2; //-384 × 288 ;
                        if(my <= 1 && my >= 0){
                            cout<<my;
                            totalPersonnes = totalPersonnes+1;
                        }
                        
                        /*if(my == 0){
                            cout<<my;
                            cout<<"On line \n";
                        }else if(my < 0 && my > -1){
                            cout<<my;
                            cout<<"Down \n";
                        }else if(my < 1 && my > 0) {
                            cout<<my;
                            cout<<"Up \n";
                        }*/
                        totalCount = "Total = " + std::to_string(totalPersonnes);

                        stringstream ssTmp;
                        ssTmp << ListObjetsActuel[i].indice;
                        string text = ssTmp.str();

                        int fontFace = CV_FONT_HERSHEY_SIMPLEX;
                        double fontScale = 0.5;
                        int thickness = 1;

                        int x = (ListObjetsActuel[i].x1 + ListObjetsActuel[i].x2)
                                / 2;
                        int y = (ListObjetsActuel[i].y1 + ListObjetsActuel[i].y2)
                                / 2;
                        Point textPosition(x, y);
                        putText(frame, totalCount, textPosition, fontFace, fontScale,
                                CV_RGB(0, 0, 255), thickness, 8);
                    }

                    //cout<<"Affichage et enregistrement des resultats images_suivi";
                    stringstream ss1;
                    ss1 << "images_suivi/" << nom_video << ".png";
                    string fileName1 = ss1.str();
                    if (!imwrite(fileName1, imageSuivi))
                        cout << "Erreur lors de l'enregistrement de line 624 " << fileName1
                             << endl;

                    stringstream ss2;
                    ss2 << "images_videos/" << nom_video << "_" << numFrameActuel
                        << ".png";
                    string fileName2 = ss2.str();
                    if (!imwrite(fileName2, frame))
                        cout << "Erreur lors de l'enregistrement de " << fileName2
                             << endl;

                    //enrégistrement de l'image de la vidéo avec les boites englobantes
                    stringstream ss3;
                    ss3 << "images_mouvement/" << nom_video << "seuil_" << seuil_detection1
                        << "_frame_" << images_video.size() - 1 << ".png";
                    string fileName3 = ss3.str();
                    if (!imwrite(fileName3, images_mouvement.back()))
                        cout << "Erreur lors de l'enregistrement de " << fileName3
                             << endl;

                    imshow("Suivi Mouvement", imageSuivi);

                    numFrameActuel++;

                    imshow(nom_video,frame);
                    imshow("Image Arriere Plan", image_arriere_plan);
                    imshow("Detection Mouvement", images_mouvement.back());

                    key = cvWaitKey(40);
                }
            }
        }
        cvDestroyAllWindows();
    }
}

// methode d'extraction de l'arriere plan
Mat bgExtraction(string vid_name, int nb_sequences)
{
    int hauteur, largeur;
    stringstream chemin_video;
    chemin_video << "videos/" << vid_name;
    string fileName = chemin_video.str();
    cout << "la video est: " << fileName << endl;
    Mat bg_pic;

    // Chargement de la video
    VideoCapture videoCapture(fileName);

    if (!videoCapture.isOpened())
    {
        cout << "Impossible de lire la video : " << vid_name << endl;
        exit(0);
    }
    else
    {
        // Dimensions des images de la video
        hauteur = videoCapture.get(CV_CAP_PROP_FRAME_HEIGHT);
        largeur = videoCapture.get(CV_CAP_PROP_FRAME_WIDTH);

        // Initialisation de l'image d'arrière-plan
        bg_pic = Mat::zeros(hauteur, largeur, CV_8UC1);

        // Vecteur contenant les séquence d'images pour la construction de l'arrière-plan
        vector<Mat> images_video;
        int countFrame = 0;
        while (true)
        {
            Mat frame, frameGray;
            //lecture du frame
            videoCapture >> frame;
            if (!frame.data)
            {
                cout << "Fin de lecture de la video" << endl;
                exit(0);
            }
            else if (countFrame < nb_sequences)
            {
                // Conversion des images en niveau de gris
                cvtColor(frame, frameGray, CV_BGR2GRAY);
                // Enregistrement de l'image dans le vecteur d'images
                images_video.push_back(frameGray);
            }
            else if (countFrame >= nb_sequences)
                break;

            countFrame++;
        }
        int nb_images = images_video.size();
        // Création de l'arrière-plan
        for (int i = 0; i < bg_pic.rows; i++)
        {
            for (int j = 0; j < bg_pic.cols; j++)
            {
                // Vecteur contenant les valeurs dans toutes les images d'un pixel donné
                vector<int> vecteur_pixel;
                // Réccupération des valeurs du pixel dans toutes les images
                for (int k = 0; k < nb_images; k++)
                {
                    int val = images_video[k].at<uchar>(i, j);
                    vecteur_pixel.push_back(val);
                }
                // Tri des valeurs du vecteur
                sort(vecteur_pixel.begin(), vecteur_pixel.end());
                // Choix de la valeur mediane
                bg_pic.at<uchar>(i, j) = vecteur_pixel[(nb_images + 1) / 2];
            }
        }
        if (bg_pic.data)
        {
            // Enregistrement de l'arrière plan
            stringstream path;
            path << "arriere_plan/" << vid_name << "_" << nb_sequences
                 << ".png";
            string fileName = path.str();
            if (!imwrite(fileName, bg_pic))
                cout << "Erreur lors de l'enregistrement de " << fileName
                     << endl;
        }
        else
        {
            cout << "Echec de l'extraction de l'arriere plan" << endl;
        }

        return bg_pic;
    }
}



// methode de detection de mouvement
vector<Mat> motionDetection(string vid_name, Mat bg_pic, int seuil)
{
    //définition des variables
    vector<Mat> images_video;
    vector<Mat> motion_pic;

    stringstream video_path;
    video_path << "videos/" << vid_name;
    string fileName = video_path.str();

    char key;
    namedWindow(vid_name, CV_WINDOW_AUTOSIZE);
    namedWindow("Image Arriere Plan", CV_WINDOW_AUTOSIZE);
    namedWindow("Detection Mouvement", CV_WINDOW_AUTOSIZE);

    //application du masque gaussien pour le lissage
    GaussianBlur(bg_pic, bg_pic, Size(5, 5), 0, 0);
    // Chargement de la video
    VideoCapture videoCapture(fileName);
    //vérification du chargement
    if (!videoCapture.isOpened())
    {
        cout << "Impossible de lire la video : " << vid_name << endl;
        exit(0);
    }
    else
    {
        // lecture des images constituant la video
        while (key != 'q' && key != 'Q')
        {
            Mat frame, frame_gray;
            videoCapture >> frame;
            //test de fin de video
            if (!frame.data)
            {
                cout << "Fin de lecture de la video" << endl;
                break;
            }
            else
            {
                // Conversion des images en niveau de gris
                cvtColor(frame, frame_gray, CV_BGR2GRAY);
                // Enregistrement de l'image dans le vecteur d'image
                images_video.push_back(frame_gray);
                //application du masque gaussien pour le lissage
                GaussianBlur(images_video.back(), images_video.back(),
                             Size(5, 5), 0, 0);
                Mat diff_images;
                // Différence entre image arriere-plan et image capturée pour détecter le mouvement
                absdiff(bg_pic, images_video.back(),
                        diff_images);
                motion_pic.push_back(diff_images);

                // Seuillage binaire pour éliminer les bruits
                threshold(motion_pic.back(), motion_pic.back(),
                          seuil, 255.0, CV_THRESH_BINARY);
                motion_pic.back() = detectionImproved(
                                        motion_pic.back());
                //détermination des boites englobantes pour les objets détectés
                vector<Shape> shapes_vector = getShape(motion_pic.back());
                drawShape(frame, shapes_vector);
                //enrégistrement des résultats
                if (motion_pic.back().data)
                {
                    // Enregistrement de l'image de mouvement
                    stringstream path;
                    path << "images_mouvement/" << vid_name << "seuil_"
                         << seuil << "_frame_" << images_video.size() - 1
                         << ".png";
                    string fileName = path.str();
                    if (!imwrite(fileName, motion_pic.back()))
                        cout << "Erreur lors de l'enregistrement de "
                             << fileName << endl;
                }
                else
                {
                    cout << "Echec de détermination du mouvement pour le frame "
                         << images_video.size() - 1 << endl;
                }

                //enrégistrement de l'image de la vidéo avec les boites englobantes
                stringstream path1;
                path1 << "images_videos/" << vid_name << "seuil_" << seuil
                      << "_frame_" << images_video.size() - 1 << ".png";
                string fileName1 = path1.str();
                if (!imwrite(fileName1, frame))
                    cout << "Erreur lors de l'enregistrement de " << fileName1
                         << endl;

                //affichages des différentes images
                imshow(vid_name, frame);
                imshow("Image Arriere Plan", bg_pic);
                imshow("Detection Mouvement", motion_pic.back());

                key = cvWaitKey(40);
            }
        }
    }
    cvDestroyAllWindows();
    return motion_pic;
}

int controlSaisie(vector<string> detailCmdUser)
/*
    Objet           : Fonction permettant de vérifier si la commande saisie par l'utilisateur est correcte
    Type de retour  : int
*/
{
    return int(detailCmdUser.size());

}//Fin de la fonction controlSaisie

void OperationToDO(vector<string> detailCmdUser)
/*
    Objet           : Fonction permettant de déterminer le module correspondant a la commande
    Type de retour  : void
*/
{
    //cout<<"In OperationToDO \n";

    int nbarg = controlSaisie(detailCmdUser);
    //cout<<detailCmdUser.at(0);
    //cout<<detailCmdUser.at(1);
    //cout<<detailCmdUser.at(2);
    //cout<<detailCmdUser.at(3);
    //cout<<detailCmdUser.at(4);
    //cout<<nbarg;
    if(nbarg==4){
        //cin >> nom_video >> nb_sequences >> seuil;

        Mat bg_pic;
        bg_pic = bgExtraction(detailCmdUser.at(1), std::stoi(detailCmdUser.at(2)));
        vector<Mat> motion_pics;
        motion_pics = motionDetection(detailCmdUser.at(1), bg_pic, std::stoi(detailCmdUser.at(3)));

        waitKey(0);
        // fin de detection de mouvements
    }else if(nbarg==5) {
        if(detailCmdUser.at(0)=="sv"){
            Mat bg_pic;
            bg_pic = bgExtraction(detailCmdUser.at(1), std::stoi(detailCmdUser.at(2)));
            //detailCmdUser.at(0).compare("pf")==0;
            motionFollowing(detailCmdUser.at(1), bg_pic, std::stoi(detailCmdUser.at(3)), std::stoi(detailCmdUser.at(4)));        
        } else if(detailCmdUser.at(0)=="cpg"){
            Mat bg_pic;
            bg_pic = bgExtraction(detailCmdUser.at(1), std::stoi(detailCmdUser.at(2)));
            //detailCmdUser.at(0).compare("pf")==0;
            motionCount(detailCmdUser.at(1), bg_pic, std::stoi(detailCmdUser.at(3)), std::stoi(detailCmdUser.at(4)));        
        }else {
            cout<<"Syntaxe ou parametres de la commande incorectes \n";
        }
    }else {
        cout<<"Syntaxe ou parametres de la commande incorectes \n";
    }

}//Fin de la fonction OperationToDO
