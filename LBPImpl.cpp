// 2D3Dtp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>

#include <bitset>
#include <algorithm>
#include <string>
#include <dirent.h>

#include <windows.h>
#include <sys/types.h>

#include <time.h>


using namespace std;

using namespace cv;


int imageDescripteur[256] = { 0 };


int compareTwoPixels(Mat image, int xc, int yc, int xp, int yp)
{
    if (image.at<uint8_t>(yc, xc) > image.at<uint8_t>(yp, xp))
        return 0;
    else
        return 1;
}

int  descripteurDunPixel(Mat image, int c, int r)
{
    int vals[8] = { 0 };

    //top left
    vals[0] = compareTwoPixels(image, c, r, c - 1, r - 1);

    //top
    vals[1] = compareTwoPixels(image, c, r, c, r -1);

    //top right
    vals[2] = compareTwoPixels(image, c, r, c + 1, r - 1);

    //right
    vals[3] = compareTwoPixels(image, c, r, c + 1, r);

    //bottom right
    vals[4] = compareTwoPixels(image, c, r, c + 1, r + 1);

    //bottom
    vals[5] = compareTwoPixels(image, c, r, c, r + 1);

    //bottom left
    vals[6] = compareTwoPixels(image, c, r, c - 1, r + 1);


    //left
    vals[7] = compareTwoPixels(image, c, r, c - 1, r);


    int puissances[8] = { 1,2,4,8,16,32,64,128 };

    int descVal = 0;
    for (int i = 0; i < 8; i++)
        descVal += vals[i] * puissances[i];

    return descVal;

}


void calculateChannelsDecripteurs(String fileName)
{
    cv::Mat image = cv::imread(fileName);

    cv::Mat imageChannels[3];

    split(image, imageChannels);

    for (int i = 0; i < 256; i++)
        imageDescripteur[i] = 0;

    for (int i = 0; i < 3; i++)
    {
        for (int r = 1; r < imageChannels[i].rows - 2; r++)
        {
            for (int c = 1; c < imageChannels[i].cols - 2; c++)
            {
                int pointScore = descripteurDunPixel(imageChannels[i], c, r);

                imageDescripteur[pointScore] += 1;

            }
        }
    }

    return;

}

void LBP(String fileName, string output, bool train, int numClass, bool isGray) {


    if (isGray)
    {
        cv::Mat image = cv::imread(fileName, IMREAD_GRAYSCALE);

        for (int i = 0; i < 256; i++)
            imageDescripteur[i] = 0;

        for (int r = 1; r < image.rows - 2; r++)
        {
            for (int c = 1; c < image.cols - 2; c++)
            {
                int pointScore = descripteurDunPixel(image, c, r);

                imageDescripteur[pointScore] += 1;

            }
        }
    }
    else {
        calculateChannelsDecripteurs(fileName);
    }

    if (train)
    {
        ofstream myfile;
        myfile.open(output, ios_base::app);


        for (int i = 0; i < 256; i++)
        {
            myfile << imageDescripteur[i] << " ";
        }

        myfile <<  numClass << " ";
    }

    return ;

}

void trainLBP(string output, string source, bool file)
{
    
        string inputDirectory = source+"\\";
        DIR* directory = opendir(inputDirectory.c_str());
        struct dirent* _dirent = NULL;
        if (directory == NULL)
        {
            printf("Cannot open Input Folder\n");
            return ;
        }
        while ((_dirent = readdir(directory)) != NULL)
        {
            std::string fileName = inputDirectory + "\\" + std::string(_dirent->d_name);

            if (fileName.find("jpg") != std::string::npos)
            {

                if(source == "trainon1" || source == "trainon2")
                    LBP(fileName.c_str(), output, true, 1,false);
                else
                    LBP(fileName.c_str(), output, true, 0,false);

            }

            
        }

    
        
}

float meanTrainImageCalc(vector<int> values)
{
    float meanTrainImage = 0;

    for (int i = 0; i < 256; i++)
    {
        meanTrainImage += values[i];
    }
    meanTrainImage = meanTrainImage / 256;

    return meanTrainImage;
}

float meanTestImageCalc()
{
    float meanTestImage = 0;

    for (int i = 0; i < 256; i++)
    {
        meanTestImage += imageDescripteur[i];
    }

    meanTestImage = meanTestImage / 256;
    return meanTestImage;
}


int hammingDistanceBetweenImages(vector<int> values)
{
    int distance = 0;
    for (int i = 0; i < 256; i++)
    {
        bitset<9> imageTestDesc(imageDescripteur[i]);
        bitset<9> imageTrainDesc(values[i]);
        bitset<9> dif = imageTestDesc ^ imageTrainDesc;
        string difString = dif.to_string();

        for (int j = 0; j < difString.size(); j++)
        {
            if (difString.at(j) == '1')
                distance++;
        }
    }
    return distance;
}

float correlationBetweenImages(vector<int> values)
{
    float sum1 = 0;
    float sum2 = 0;

    float distance = 0;

    float meanTestImage = meanTestImageCalc();
    float meanTrainImage = meanTrainImageCalc(values);

    for (int i = 0; i < 256 ;  i++)
    {
        sum1 += (imageDescripteur[i] - meanTestImage)*(values[i] - meanTrainImage);
        sum2 += pow(imageDescripteur[i] - meanTestImage,2) * pow(values[i] - meanTrainImage,2);
    }

    distance = sum1 / sqrt(sum2);
  
    return distance;
}

float chiSquareBetweenImages(vector<int> values)
{
    float distance = 0;

    for (int i = 0; i < 256; i++)
    {
        distance += pow(imageDescripteur[i] - values[i], 2) / imageDescripteur[i];
    }

    return distance;
}

float intersactionBetweenImages(vector<int> values)
{
    float distance = 0;

    for(int i = 0; i<256;i++)
    {
        imageDescripteur[i] < values[i] ? distance += imageDescripteur[i] : distance += values[i];
    }

    return distance;
}

int bhattacharyyaDistanceBetweenImages(vector<int> values)
{
    float distance = 0;
    float sum = 0;

    float meanTestImage = meanTestImageCalc();
    float meanTrainImage = meanTrainImageCalc(values);

    for (int i = 0; i < 256; i++)
    {
        sum += sqrt(imageDescripteur[i] * values[i]);
    }

    distance = sum / (sqrt(meanTestImage * meanTrainImage * pow(256, 2)));

    distance = 1 - distance;

    return distance;
}

float sumSquaredDifference(vector<int> values)
{
    float distance = 0;
    int descriptorIndex = 0;

    for (auto val : values)
    {
        if (descriptorIndex < 256)
        {
            distance += pow(val - imageDescripteur[descriptorIndex],2);
            descriptorIndex++;
        }
        else
            break;
    }


    return distance;
}


float sumAbsDiffrence(vector<int> values)
{
    float distance = 0;
    int descriptorIndex = 0;

    for (auto val : values)
    {
        if (descriptorIndex < 256)
        {
            distance += abs(val - imageDescripteur[descriptorIndex]);
            descriptorIndex++;
        }
        else
            break;
    }
    return distance;
}

float calculateDitanceBetweenImageDesc(vector<int> values, int mesure)
{
    float distance = 0;
    switch (mesure)
    {
    case  0 :
        distance = sumAbsDiffrence(values);
        break;
    case 1 :
        distance = sumSquaredDifference(values);
        break;
    case 2 :
        distance = correlationBetweenImages(values);
        break;
    case 3 :
        distance = chiSquareBetweenImages(values);
        break;
    case 4 : 
        distance = intersactionBetweenImages(values);
        break;
    case 5 :
        distance = bhattacharyyaDistanceBetweenImages(values);
        break;
    case 6 :
        distance = hammingDistanceBetweenImages(values);
        break;
    default:
        distance = sumAbsDiffrence(values);
    }
    return distance;
    
}

int testImage(String imageFile, String source, int mesure)
{
    float bestDistance = -1;
    int bestClass = 0;

   
    LBP(imageFile, "", false, 0, false);

    ifstream myfile;
    
    if(source == "testOn1" || source == "testOff1")
        myfile.open("train1.txt");
    else
        myfile.open("train2.txt");


    string line;

    vector<int> valeurs;

    while (getline(myfile, line,' '))
    {
        try
        {
            int val = stoi(line);
            valeurs.push_back(val);
        }catch (const std::exception&){
            cout << "Exception here" << endl;
        }

        if (valeurs.size() >= 257)
        {
            float calculatedDistance = calculateDitanceBetweenImageDesc(valeurs,mesure);

            if (bestDistance == -1 || bestDistance > calculatedDistance)
            {
                bestDistance = calculatedDistance;
                bestClass = valeurs.back();   
            }
            valeurs.clear();
        }

    }
    myfile.close();
    return bestClass;
}

int testLBP(string source, int classToPredict, int mesure)
{

    int imageCoount = 0;

    string inputDirectory = source + "\\";
    DIR* directory = opendir(inputDirectory.c_str());
    struct dirent* _dirent = NULL;
    if (directory == NULL)
    {
        printf("Cannot open Input Folder\n");
        return 0;
    }


    int nbrGoodPrediction = 0;
    int nbrBadPrediction = 0;

    while ((_dirent = readdir(directory)) != NULL)
    {
        std::string fileName = inputDirectory + "\\" + std::string(_dirent->d_name);

        if (fileName.find("jpg") != std::string::npos)
        {
            int classPredicted = testImage(fileName.c_str(), source, mesure);
            classPredicted == classToPredict ? nbrGoodPrediction++ : nbrBadPrediction++;  
            imageCoount++;
            cout << imageCoount << endl;
        }
    }

   

    return nbrGoodPrediction;
}


int main()
{
    clock_t tStart = clock();

    trainLBP("train1.txt", "trainon1", false);
    trainLBP("train1.txt", "trainoff1", false);

    cout << "end traing, done in " << (double)(clock() - tStart) / CLOCKS_PER_SEC <<endl;
    
    
    

   

    int goodPredictions = 0;
   
    //CHI-S
    tStart = clock();

    goodPredictions = testLBP("testOn1", 1, 3);
    goodPredictions = goodPredictions + testLBP("testOff1", 0, 3);

    cout << "end test 1,  done in " << (double)(clock() - tStart) / CLOCKS_PER_SEC << endl;

    cout << "Resultat test premiere dataset, CHI-S" << endl;

    cout << " bonne prediction :  " << goodPredictions << endl;
    cout << " mauvase predictione : " << 1500 - goodPredictions << endl;




    goodPredictions = 0;
    //HAMMING
    tStart = clock();

    goodPredictions = testLBP("teston1", 1,6);
    goodPredictions = goodPredictions + testLBP("testoff1", 0,6);

    cout << "end test 1,  done in " << (double)(clock() - tStart) / CLOCKS_PER_SEC << endl;

    cout << "resultat test premiere dataset, Hamming" << endl;
    
    cout << " bonne prediction :  " << goodPredictions << endl;
    cout << " mauvase predictione : " << 1500 - goodPredictions << endl;


   





 
}


