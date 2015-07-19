import cv2, os, fnmatch, shutil, re, sys, getopt
import numpy as np
import scipy as sc
from PIL import Image
from subprocess import call


def main():
    refDir = getArgs(sys.argv[1:])
    imagesDir = "./images"
    matchesDir = "./matches"

    detects = recognizeFaces_LBPH(refDir, imagesDir, matchesDir)
    detects = sorted(detects, key=lambda k: k['confidence'])
    for detected_face in detects:
        if detected_face['confidence'] < 100:
            foundImage = imagesDir+"/subject"+str(detected_face['search_label'])+'.jpeg'
            img = cv2.imread(foundImage)
            imgNumpy = np.array(img, 'uint8')
            cv2.imshow("Match",imgNumpy)
            cv2.waitKey(0)




def usage():
    print 'USAGE : ./' + sys.argv[0] + ' -r <Reference Directory>'
    sys.exit(2)


def getArgs(argv):
    refDir = ''
    try:
        opts, args = getopt.getopt(argv, "r:", ["ref="])
    except getopt.GetoptError:
        usage()

    for opt, arg in opts:
        if opt in ("-r", "--ref"):
            refDir = arg

    if not refDir:
        usage()

    if not os.path.exists(refDir):
        print "Error : Missing Reference Directory Path\n"
        usage()

    return refDir


def getImages(path):
    renameDir(path)
    output = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            pattern = re.compile(".*jpg|.*png|.*jpeg$")
            if pattern.match(filename):
                output.append(os.path.join(root, filename))
    return output


def renameDir(dirPath):
    index=0
    for root, dirs, filenames in os.walk(dirPath):
        for filename in filenames:
            pattern = re.compile("^\.|subject\d+\.")
            if pattern.match(filename):
                continue
            index= index + 1
            name, ext = os.path.splitext(filename)
            ext = ".jpeg"
            destFile = dirPath+"/"+"subject"+str(index)+ext
            while os.path.exists(destFile):
                index = index+1
                destFile = dirPath+"/"+"subject"+str(index)+ext
            os.rename(dirPath+"/"+filename,destFile)

def cleanDir(dirPath):
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath + "/" + fileName)


def cleanDS_Store(dirPath):
    call(["find", dirPath + "/ -type f -name \".DS_Store\" -delete"])


def getLabel(path):
    return int(os.path.split(path)[1].split(".")[0].replace("subject", ""))


def getFaces_HaarCascade(imgPath):
    haarFaceCascade = 'cascades/haarcascade_frontalface_default.xml'

    faceCascade = cv2.CascadeClassifier(haarFaceCascade)

    # Convert to Grayscale
    imgGray = Image.open(imgPath).convert('L')

    # covert to numpy array
    imgNumpy = np.array(imgGray, 'uint8')

    # Detect all faces using haar face cascade in Image's Numpy array
    faces = faceCascade.detectMultiScale(imgNumpy)

    output = []
    for face in faces:
        x, y, w, h = face
        array = imgNumpy[y: y + h, x: x + w]
        array = cv2.resize(array, (100, 100))
        array = cv2.equalizeHist(array)
        output.append(array)
        # cv2.imshow("Face",array)
        # cv2.waitKey(0)

    return output


def trainFaceRecognizer_LBPH(recognizer, path):
    images = []
    labels = []
    for imgPath in getImages(path):
        label = getLabel(imgPath)
        faces = getFaces_HaarCascade(imgPath)
        for face in faces:
            images.append(face)
            labels.append(label)
    recognizer.train(images, np.array(labels))


def recognizeFaces_LBPH(refDir, imagesDir, matchesDir):
    recognizer = cv2.createLBPHFaceRecognizer()

    trainFaceRecognizer_LBPH(recognizer, refDir)

    searchImages = getImages(imagesDir)

    outputs = []
    for imgPath in searchImages:
        faces = getFaces_HaarCascade(imgPath)
        for face in faces:
            output = {}
            output["ref_label"], output["confidence"] = recognizer.predict(face)
            output["search_label"] = getLabel(imgPath)
            outputs.append(output)
    return outputs


if __name__ == '__main__':
    main()
