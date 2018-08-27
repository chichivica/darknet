#include "network.h"
#include "image.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "classifier.h"
#include "demo.h"
#include <sys/time.h>

#define DEMO 1

#ifdef OPENCV

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static network *net;
static image buff [3];
static image buff_letter[3];
static int buff_index = 0;
static CvCapture * cap;
static IplImage  * ipl;
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier = .5;
static int running = 0;

static int demo_frame = 3;
static int demo_index = 0;
static float **predictions;
static float *avg;
static int demo_done = 0;
static int demo_total = 0;
double demo_time;

network *classifier_net; // ******
int classifier_version = 2; // 1 or 2
int flag_detection = 0;  // flag is set if the target was detected
int predict_class(image im, network *classifier_net, float *pprob);
void draw_box_width_relative_label(image im, box bbox, int linewidth, double *rgb, char* labelstr, image **alphabet);

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter);

int size_network(network *net)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            count += l.outputs;
        }
    }
    return count;
}

void remember_network(network *net)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(predictions[demo_index] + count, net->layers[i].output, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
}

detection *avg_predictions(network *net, int *nboxes)
{
    int i, j;
    int count = 0;
    fill_cpu(demo_total, 0, avg, 1);
    for(j = 0; j < demo_frame; ++j){
        axpy_cpu(demo_total, 1./demo_frame, predictions[j], 1, avg, 1);
    }
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(l.output, avg + count, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
    detection *dets = get_network_boxes(net, buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, nboxes, 1);
    return dets;
}

void *detect_in_thread(void *ptr)
{
    running = 1;
    float nms = .4;

    layer l = net->layers[net->n-1];
    float *X = buff_letter[(buff_index+2)%3].data;
    network_predict(net, X);

    /*
       if(l.type == DETECTION){
       get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
       } else */
    remember_network(net);
    detection *dets = 0;
    int nboxes = 0;
    dets = avg_predictions(net, &nboxes);


    /*
       int i,j;
       box zero = {0};
       int classes = l.classes;
       for(i = 0; i < demo_detections; ++i){
       avg[i].objectness = 0;
       avg[i].bbox = zero;
       memset(avg[i].prob, 0, classes*sizeof(float));
       for(j = 0; j < demo_frame; ++j){
       axpy_cpu(classes, 1./demo_frame, dets[j][i].prob, 1, avg[i].prob, 1);
       avg[i].objectness += dets[j][i].objectness * 1./demo_frame;
       avg[i].bbox.x += dets[j][i].bbox.x * 1./demo_frame;
       avg[i].bbox.y += dets[j][i].bbox.y * 1./demo_frame;
       avg[i].bbox.w += dets[j][i].bbox.w * 1./demo_frame;
       avg[i].bbox.h += dets[j][i].bbox.h * 1./demo_frame;
       }
    //copy_cpu(classes, dets[0][i].prob, 1, avg[i].prob, 1);
    //avg[i].objectness = dets[0][i].objectness;
    }
     */

    if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n");
    image display = buff[(buff_index+2) % 3];

    //draw_detections(display, dets, nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes);

    //--------------- ***
    //int *indexes = calloc(top, sizeof(int));
    char **names = demo_names;
    int money_class_index = 0;    

    for(int i = 0; i < nboxes; ++i){
        //char labelstr[4096] = {0};
        //int class = -1;        

        double thresh = 0.05;

        int any_class = 0;
        int class_index_max_prob = -1; // 
        float max_prob = 0;        

        for(int j = 0; j < l.classes; ++j) {

            //printf("box i=%d, class j=%d, prob=%5.2f%%\n", i, j, dets[i].prob[j]);
            if (dets[i].prob[j] > thresh) {
              any_class = 1;
              // use native (yolo) classifier:
              if (dets[i].prob[j] > max_prob) {
                max_prob = dets[i].prob[j];
                class_index_max_prob = j;
              }
            }
        }

        // -----
        // Insertion of an external classifier (tiny, darknet19 or else)
        if (any_class) {

            printf("box i=%d:\n", i);

            double det_x = dets[i].bbox.x;
            double det_y = dets[i].bbox.y;
            double det_w = dets[i].bbox.w;
            double det_h = dets[i].bbox.h;
            //printf("det: %f %f %f %f\n", det_x, det_y, det_w, det_h);
            image im_box = 
                get_piece_of_image_rectangle(display, det_x, det_y, det_w, det_h);

            float prob_classifier;            
            int class_index = predict_class(im_box, classifier_net, &prob_classifier);
            printf("YOLO prediction: class=%d (%s) [%5.2f%%];\n", class_index_max_prob, names[class_index_max_prob], max_prob*100);
            printf("classifier net:  class=%d (%s) [%5.2f%%];\n", class_index, names[class_index], prob_classifier*100);

            float thresh_target = 0.6;
            int yolo_detects_target = 0;
            int classifier_detects_target = 0;

            if (class_index_max_prob == money_class_index) {
              // if yolo detected money
              yolo_detects_target = 1;              
            }

            if ((class_index == money_class_index) && (prob_classifier >= thresh_target)) {
              // if classifier detected money
              classifier_detects_target = 1;              
            }

            if (classifier_detects_target) {
              printf("* classifier detected money in the box %f %f %f %f\n", det_x, det_y, det_w, det_h);
            }            

            //if (yolo_detects_target || classifier_detects_target) {

            int flag_draw = 0;
            int linewidth = 4;
            double color[3];
            char labelstr[256] = {0};

            if (classifier_detects_target && classifier_detects_target) {
              color[0] = 0.0; color[1] = 0.6; color[2] = 0.99; // blue
              sprintf(labelstr, "%.2lf", prob_classifier);     
              printf("<<< Both ones detects money >>>\n");
              flag_detection = 2;        
            }            

            if (yolo_detects_target && (!classifier_detects_target)) {
              color[0] = 0.8; color[1] = 0.1; color[2] = 0.2;  // red
              sprintf(labelstr, "err %.2lf", prob_classifier);
              if (flag_detection < 1) flag_detection = 1;
            }

            if (!yolo_detects_target) {
              color[0] = 0.1; color[1] = 0.8; color[2] = 0.0; // green
              sprintf(labelstr, "-");
            }

            /*
            switch (classifier_version) {
              case 1: flag_draw = 1; break;  //
              case 2: if (yolo_detects_target) flag_draw = 1; break;
            }
            if  (flag_draw && (yolo_detects_target || classifier_detects_target)) {
              draw_box_width_relative_label(display, dets[i].bbox, linewidth, color, labelstr, demo_alphabet);
            } 
            */    

            draw_box_width_relative_label(display, dets[i].bbox, linewidth, color, labelstr, demo_alphabet);

        }
    }
    //---------------
    free_detections(dets, nboxes);

    demo_index = (demo_index + 1)%demo_frame;
    running = 0;
    return 0;
}

void *fetch_in_thread(void *ptr)
{
    int status = fill_image_from_stream(cap, buff[buff_index]);
    letterbox_image_into(buff[buff_index], net->w, net->h, buff_letter[buff_index]);
    if(status == 0) demo_done = 1;
    return 0;
}

void *display_in_thread(void *ptr)
{
    show_image_cv(buff[(buff_index + 1)%3], "Demo", ipl);
    int c = cvWaitKey(1);
    if (c != -1) c = c%256;
    if (c == 27) {
        demo_done = 1;
        return 0;
    } else if (c == 82) {
        demo_thresh += .02;
    } else if (c == 84) {
        demo_thresh -= .02;
        if(demo_thresh <= .02) demo_thresh = .02;
    } else if (c == 83) {
        demo_hier += .02;
    } else if (c == 81) {
        demo_hier -= .02;
        if(demo_hier <= .0) demo_hier = .0;
    }
    return 0;
}

void *display_loop(void *ptr)
{
    while(1){
        display_in_thread(0);
    }
}

void *detect_loop(void *ptr)
{
    while(1){
        detect_in_thread(0);
    }
}

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    //demo_frame = avg_frames;
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    printf("Demo\n");
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    //-------------
    //char *classifier_cfgfile = "cfg/money_tiny_test.cfg";
    //char *classifier_weightfile = "../money_tiny.weights"; 
    char *classifier_cfgfile = "cfg/money_darknet19.cfg";
    char *classifier_weightfile = "../money_darknet19.weights";     
    classifier_net = load_network(classifier_cfgfile, classifier_weightfile, 0);
    set_batch_network(classifier_net, 1);
    flag_detection = 0;
    //-------------
    pthread_t detect_thread;
    pthread_t fetch_thread;

    srand(2222222);

    int i;
    demo_total = size_network(net);
    predictions = calloc(demo_frame, sizeof(float*));
    for (i = 0; i < demo_frame; ++i){
        predictions[i] = calloc(demo_total, sizeof(float));
    }
    avg = calloc(demo_total, sizeof(float));

    if(filename){
        printf("video file: %s\n", filename);
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);

        if(w){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
        }
        if(h){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
        }
        if(frames){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
        }
    }

    if(!cap) error("Couldn't connect to webcam.\n");

    double video_fps = cvGetCaptureProperty(cap, CV_CAP_PROP_FPS);
    double video_width = cvGetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH);
    double video_height = cvGetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT);

    CvVideoWriter *writer;
    if (prefix){
        writer = cvCreateVideoWriter(prefix, CV_FOURCC('D', 'I', 'V', 'X'), video_fps,
                                     cvSize((int) video_width, (int) video_height), 1);
    }


    buff[0] = get_image_from_stream(cap);
    buff[1] = copy_image(buff[0]);
    buff[2] = copy_image(buff[0]);
    buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[2] = letterbox_image(buff[0], net->w, net->h);
    ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);

    int count = 0;
    if(!prefix){
        cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
        if(fullscreen){
            cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        } else {
            cvMoveWindow("Demo", 0, 0);
            cvResizeWindow("Demo", 1352, 1013);
        }
    }

    demo_time = what_time_is_it_now();

    while(!demo_done){
        buff_index = (buff_index + 1) %3;
        if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
        if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");

        fps = 1./(what_time_is_it_now() - demo_time);
        demo_time = what_time_is_it_now();

        if(!prefix){
            display_in_thread(0);
        }else{
            IplImage *ipl = image_to_ipl(buff[(buff_index + 1)%3]);
            cvWriteFrame(writer, ipl);
            cvReleaseImage(&ipl);


//            char name[256];
//            sprintf(name, "%s_%08d", prefix, count);
//            save_image(buff[(buff_index + 1)%3], name);
        }
        pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);
        ++count;
    }

    cvReleaseVideoWriter(&writer);
    cvReleaseCapture(&cap);

    // -----------------
    // *** Split output videos into two directories
    if (prefix) {
      char cmd[1024];
      sprintf(cmd, "mv %s %d\n", prefix, flag_detection);
      printf(cmd);
      system(cmd);
    }

}

/*
   void demo_compare(char *cfg1, char *weight1, char *cfg2, char *weight2, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
   {
   demo_frame = avg_frames;
   predictions = calloc(demo_frame, sizeof(float*));
   image **alphabet = load_alphabet();
   demo_names = names;
   demo_alphabet = alphabet;
   demo_classes = classes;
   demo_thresh = thresh;
   demo_hier = hier;
   printf("Demo\n");
   net = load_network(cfg1, weight1, 0);
   set_batch_network(net, 1);
   pthread_t detect_thread;
   pthread_t fetch_thread;

   srand(2222222);

   if(filename){
   printf("video file: %s\n", filename);
   cap = cvCaptureFromFile(filename);
   }else{
   cap = cvCaptureFromCAM(cam_index);

   if(w){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
   }
   if(h){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
   }
   if(frames){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
   }
   }

   if(!cap) error("Couldn't connect to webcam.\n");

   layer l = net->layers[net->n-1];
   demo_detections = l.n*l.w*l.h;
   int j;

   avg = (float *) calloc(l.outputs, sizeof(float));
   for(j = 0; j < demo_frame; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));

   boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
   probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
   for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes+1, sizeof(float));

   buff[0] = get_image_from_stream(cap);
   buff[1] = copy_image(buff[0]);
   buff[2] = copy_image(buff[0]);
   buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
   buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
   buff_letter[2] = letterbox_image(buff[0], net->w, net->h);
   ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);

   int count = 0;
   if(!prefix){
   cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
   if(fullscreen){
   cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
   } else {
   cvMoveWindow("Demo", 0, 0);
   cvResizeWindow("Demo", 1352, 1013);
   }
   }

   demo_time = what_time_is_it_now();

   while(!demo_done){
buff_index = (buff_index + 1) %3;
if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
if(!prefix){
    fps = 1./(what_time_is_it_now() - demo_time);
    demo_time = what_time_is_it_now();
    display_in_thread(0);
}else{
    char name[256];
    sprintf(name, "%s_%08d", prefix, count);
    save_image(buff[(buff_index + 1)%3], name);
}
pthread_join(fetch_thread, 0);
pthread_join(detect_thread, 0);
++count;
}
}
*/
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

