#include "darknet.h"
#include <dirent.h>
#define _GNU_SOURCE
#include <stdio.h>
#include <libgen.h>
#include <layer.h>

#include "image.h"

static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};


void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    char *backup_directory = option_find_str(options, "backup", "/backup/");

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network **nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    data train, buffer;

    layer l = net->layers[net->n - 1];

    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = get_base_args(net);
    args.coords = l.coords;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    //args.type = INSTANCE_DATA;
    args.threads = 64;

    pthread_t load_thread = load_data(args);
    double time;
    int count = 0;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net->max_batches){
        if(l.random && count++%10 == 0){
            printf("Resizing\n");
            int dim = (rand() % 10 + 10) * 32;
            if (get_current_batch(net)+200 > net->max_batches) dim = 608;
            //int dim = (rand() % 4 + 16) * 32;
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            #pragma omp parallel for
            for(i = 0; i < ngpus; ++i){
                resize_network(nets[i], dim, dim);
            }
            net = nets[0];
        }
        time=what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        /*
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[10] + 1 + k*5);
           if(!b.x) break;
           printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
           }
         */
        /*
           int zz;
           for(zz = 0; zz < train.X.cols; ++zz){
           image im = float_to_image(net->w, net->h, 3, train.X.vals[zz]);
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[zz] + k*5, 1);
           printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
           draw_bbox(im, b, 1, 1,0,0);
           }
           show_image(im, "truth11");
           cvWaitKey(0);
           save_image(im, "truth11");
           }
         */

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);

        time=what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);
        printf("%ld: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, i*imgs);
        if(i%100==0){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
        }
        if(i%1000==0 || (i < 1000 && i%100 == 0)){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}


static int get_coco_image_id(char *filename)
{
    char *p = strrchr(filename, '/');
    char *c = strrchr(filename, '_');
    if(c) p = c;
    return atoi(p+1);
}

static void print_cocos(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h)
{
    int i, j;
    int image_id = get_coco_image_id(image_path);
    for(i = 0; i < num_boxes; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx = xmin;
        float by = ymin;
        float bw = xmax - xmin;
        float bh = ymax - ymin;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, dets[i].prob[j]);
        }
    }
}

void print_detector_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2. + 1;

        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void print_imagenet_detections(FILE *fp, int id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            int class = j;
            if (dets[i].prob[class]) fprintf(fp, "%d %d %f %f %f %f %f\n", id, j+1, dets[i].prob[class],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void validate_detector_flip(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 2);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if(0==strcmp(type, "coco")){
        if(!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        if(!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if(!outfile) outfile = "comp4_det_test_";
        fps = calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .005;
    float nms = .45;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    image input = make_image(net->w, net->h, net->c*2);

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    //args.type = IMAGE_DATA;
    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            copy_cpu(net->w*net->h*net->c, val_resized[t].data, 1, input.data, 1);
            flip_image(val_resized[t]);
            copy_cpu(net->w*net->h*net->c, val_resized[t].data, 1, input.data + net->w*net->h*net->c, 1);

            network_predict(net, input.data);
            int w = val[t].w;
            int h = val[t].h;
            int num = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &num, 1);
            if (nms) do_nms_sort(dets, num, classes, nms);
            if (coco){
                print_cocos(fp, path, dets, num, classes, w, h);
            } else if (imagenet){
                print_imagenet_detections(fp, i+t-nthreads+1, dets, num, classes, w, h);
            } else {
                print_detector_detections(fps, id, dets, num, classes, w, h);
            }
            free_detections(dets, num);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR);
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
}


void validate_detector(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if(0==strcmp(type, "coco")){
        if(!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        if(!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if(!outfile) outfile = "comp4_det_test_";
        fps = calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }


    int m = plist->size;
    int i=0;
    int t;

    float thresh = .005;
    float nms = .45;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    //args.type = IMAGE_DATA;
    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            int nboxes = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &nboxes, 1);
            if (nms) do_nms_sort(dets, nboxes, classes, nms);
            if (coco){
                print_cocos(fp, path, dets, nboxes, classes, w, h);
            } else if (imagenet){
                print_imagenet_detections(fp, i+t-nthreads+1, dets, nboxes, classes, w, h);
            } else {
                print_detector_detections(fps, id, dets, nboxes, classes, w, h);
            }
            free_detections(dets, nboxes);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR);
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
}

void validate_detector_recall(char *datacfg, char *cfgfile, char *weightfile,  float thresh)
{
    list *options = read_data_cfg(datacfg);

    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    char *validation_list = option_find_str(options, "valid", "data/coco_val_5k.list");
    list *plist = get_paths(validation_list);
    char **paths = (char **)list_to_array(plist);

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));


    layer l = net->layers[net->n-1];

    int j, k;

    int m = plist->size;
    int i=0;

    float iou_thresh = .5;
    float nms = .45;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    int false_positive = 0;
    int false_negative = 0;
    int true_positive = 0;
    int incorrect_label = 0;

    int total_objects = 0;
    int letter_box = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
//        image sized = resize_image(orig, net->w, net->h); letter_box = 0;
        image sized = letterbox_image(orig, net->w, net->h); letter_box = 1;
        char *id = basecfg(path);
        network_predict(net, sized.data);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, orig.w, orig.h, thresh, .5, 0, 1, &nboxes, letter_box);
        if (nms) do_nms_obj(dets, nboxes, l.classes, nms);
//        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

        char labelpath[4096];
        find_replace(path, "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);

//        image **alphabet = load_alphabet();
//        draw_detections(orig, dets, nboxes, thresh, names, alphabet, l.classes);
//        cvNamedWindow("predictions", CV_WINDOW_NORMAL);
//        show_image(orig, "predictions");
//        cvWaitKey(0);
//        cvDestroyAllWindows();

        for (k = 0; k < nboxes; ++k) { //across detections
            int dets_k_class = max_index(dets[k].prob, dets[k].classes);
            double dets_k_prob = dets[k].prob[dets_k_class];

            if (dets[k].objectness > thresh) {
//                int dets_k_class = max_index(dets[k].prob, dets[k].classes);

                ++proposals;
                float best_iou = 0;
                int best_fit_index = -1;

                for (j = 0; j < num_labels; ++j) {
                    box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
                    float iou = box_iou(dets[k].bbox, t);
                    if (iou > best_iou) {
                        best_iou = iou;
                        best_fit_index = j;
                    }
                }
                if (best_iou < iou_thresh) {
                    false_positive++;
//                    printf("false positive: %s\n", path);
                } else if (truth[best_fit_index].id != dets_k_class) {
                    incorrect_label++;
                }


            }
        }
        for (j = 0; j < num_labels; ++j) { //over ground truth objects
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            float best_iou = 0;
            int best_fit_index = -1;
            total_objects ++;
            for(k = 0; k < nboxes; ++k){ //over detections
                float iou = box_iou(dets[k].bbox, t);
                int dets_k_class = max_index(dets[k].prob, dets[k].classes);
                double dets_k_prob = dets[k].prob[dets_k_class];

                if(dets[k].objectness > thresh && iou > best_iou){
                    best_iou = iou;
                    best_fit_index = k;
                }
            }
            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;
                int dets_k_class = max_index(dets[best_fit_index].prob, dets[best_fit_index].classes);
                if (dets_k_class == truth[j].id){
                    true_positive ++;
                }

            } else {
                false_negative ++;
            }
        }

        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total,
                (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
        printf("false positive: %d incorrect label: %d false negative: %d true positive %d [total: %d]\n",
               false_positive,
               incorrect_label,
               false_negative,
               true_positive,
               total_objects
        );

        free(id);
        free_image(orig);
        free_image(sized);
    }
}

const char *get_filename_ext(const char *filename) {
    const char *dot = strrchr(filename, '.');
    if(!dot || dot == filename) return "";
    return dot + 1;
}


void write_coordinates_to_json_file(
            char *json_file,
            image im,
            detection *dets,
            int nboxes,
            float thresh,
            char **names,
            int classes)
{
    /* 
    Save the coordinates of boxes to json file.                
    ./darknet detector folder cfg/voc.data cfg/yolov3-voc.cfg yolov3.weights /ram/tmp/img
    */
    int debug = 1;
    int comma = 0;

    if (debug) {
        printf("file: %s\n", json_file);
        printf("classes = %d\n", classes);
        printf("nboxes = %d\n", nboxes);
        printf("thresh = %f\n", thresh);
    }

    FILE *fp = fopen(json_file, "w");
    fprintf(fp, "{\n");
    fprintf(fp, "  \"image\": 1,\n");
    fprintf(fp, "  \"samples\": [\n");

    for (int i = 0; i < nboxes; ++i) {

        for(int j = 0; j < classes; ++j) {

            float prob = dets[i].prob[j];

            if (prob > thresh) {

                int class = j;
                box b = dets[i].bbox;

                float xmin = b.x - b.w / 2.0;
                float xmax = b.x + b.w / 2.0;
                float ymin = b.y - b.h / 2.0;
                float ymax = b.y + b.h / 2.0;

                if (xmin < 0) xmin = 0;
                if (xmax > 1.0) xmax = 1.0;
                if (ymin < 0) ymin = 0;
                if (ymax > 1.0) ymax = 1.0;

                if (debug) {
                    printf("box %d, class %d: prob = %f\n", i, j, prob);
                    printf("x=%f, y=%f, w=%f, h=%f\n", b.x, b.y, b.w, b.h);
                    printf("xmin = %f\n", xmin);
                    printf("xmax = %f\n", xmax);
                    printf("ymin = %f\n", ymin);
                    printf("ymax = %f\n", ymax);
                }

                if (comma) fprintf(fp, ",\n");
                else comma = 1;
                fprintf(fp, "     {\n");
                fprintf(fp, "        \"class\": %d,\n", class);
                fprintf(fp, "        \"flags\": 0,\n");
                fprintf(fp, "        \"rect\": [\n");
                fprintf(fp, "          %f,\n", xmin);
                fprintf(fp, "          %f,\n", xmax);
                fprintf(fp, "          %f,\n", ymin);
                fprintf(fp, "          %f\n",  ymax);
                fprintf(fp, "        ]\n");
                fprintf(fp, "     }");

            }
        }
    }

    fprintf(fp, "\n   ]\n");
    fprintf(fp, "}\n");
    fclose(fp);
}


void folder_detector(char *datacfg, char *cfgfile, char *weightfile, char *folder_name, float thresh, float hier_thresh,
                     char *outfile, int fullscreen) {
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    float nms=.45;
    DIR *dir;
    struct dirent *ent;

    char folder_name_slash[1024];

    if (folder_name[strlen(folder_name) - 1] != '/'){
        sprintf(folder_name_slash, "%s/", folder_name);
    } else {
        sprintf(folder_name_slash, "%s", folder_name);
    }

    if ((dir = opendir(folder_name)) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir(dir)) != NULL) {
            char *file_ext = get_filename_ext(ent->d_name);
            char file_full_name[1024];
            char inference_file[1024];
            char json_file_full_path[1024];
            inference_file[0] = '\0';
            sprintf(inference_file, "%s%s_inf", folder_name_slash, ent->d_name);
            sprintf(json_file_full_path, "%s%s.json", folder_name_slash, ent->d_name);

            if (strcmp(file_ext, "jpg") == 0 ||
                strcmp(file_ext, "png") == 0) {
                printf("processing: %s\n", (ent->d_name));
                sprintf(file_full_name, "%s%s", folder_name_slash, ent->d_name);

                image im = load_image_color(file_full_name, 0, 0);
                image sized = letterbox_image(im, net->w, net->h);
                layer l = net->layers[net->n - 1];

                float *X = sized.data;
                time = what_time_is_it_now();
                network_predict(net, X);
                printf("%s: Predicted in %f seconds.\n", ent->d_name, what_time_is_it_now() - time);
                int nboxes = 0;
                detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, 1);
                //printf("%d\n", nboxes);
                //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
                if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
                draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);

                printf("saving: %s\n", inference_file);
                save_image(im, inference_file);

                // write-coordinates
   				printf("Write coordinates to json file: %s\n", json_file_full_path);
                write_coordinates_to_json_file(
                    json_file_full_path, im, dets, nboxes, thresh, names, l.classes);
   				// ------------
                free_detections(dets, nboxes);
                free_image(im);
                free_image(sized);
            }
        }
        closedir(dir);
    } else {
        /* could not open directory */
        perror("could not open provided directory");
        exit(1);
    }

}

void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    //----------
    char *classifier_cfgfile = "cfg/money_tiny_test.cfg";
    char *classifier_weightfile = "../money_tiny_459.weights"; 
    network *classifier_net = load_network(classifier_cfgfile, classifier_weightfile, 0);
    set_batch_network(classifier_net, 1);
    int top = 1;
    int *indexes = calloc(top, sizeof(int));
    //----------

    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    float nms=.45;
    int letter_box = 0;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
//        image sized = resize_image(im, net->w, net->h); letter_box = 0;
        image sized = letterbox_image(im, net->w, net->h); letter_box = 1;
        //image sized = resize_image(im, net->w, net->h);
        //image sized2 = resize_max(im, net->w);
        //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        //resize_network(net, sized.w, sized.h);
        layer l = net->layers[net->n-1];


        float *X = sized.data;
        time=what_time_is_it_now();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letter_box);
        //printf("%d\n", nboxes);
        if (nms) do_nms_obj(dets, nboxes, l.classes, nms);
//        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);        

        // ------------ ***
        for(int i = 0; i < nboxes; ++i){
            //char labelstr[4096] = {0};
            //int class = -1;

            int any_class = 0;
            for(int j = 0; j < l.classes; ++j) {
                if (dets[i].prob[j] > thresh) any_class = 1;
            }

            if (any_class) {
                double det_x = dets[i].bbox.x;
                double det_y = dets[i].bbox.y;
                double det_w = dets[i].bbox.w;
                double det_h = dets[i].bbox.h;
                image im_box = 
                    get_piece_of_image_rectangle(im, det_x, det_y, det_w, det_h);

                image r_box = letterbox_image(im_box, classifier_net->w, classifier_net->h);
                float *X_box = r_box.data;
                float *predictions = network_predict(classifier_net, X_box);
                if(classifier_net->hierarchy) hierarchy_predictions(predictions, classifier_net->outputs, classifier_net->hierarchy, 1, 1);
                top_k(predictions, classifier_net->outputs, top, indexes);
                for(i = 0; i < top; ++i){
                    int index = indexes[i];
                    //if(net->hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net->hierarchy->parent[index] >= 0) ? names[net->hierarchy->parent[index]] : "Root");
                    //else printf("%s: %f\n",names[index], predictions[index]);
                    printf("classifier: %5.2f%%: %s\n", predictions[index]*100, names[index]);
                }
                if(r_box.data != im_box.data) free_image(r_box);
                free_image(im_box);
            }
        }
        // --------
        draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
        free_detections(dets, nboxes);
        if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions");
#ifdef OPENCV
            cvNamedWindow("predictions", CV_WINDOW_NORMAL);
            if(fullscreen){
                cvSetWindowProperty("predictions", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
            }
            show_image(im, "predictions");
            cvWaitKey(0);
            cvDestroyAllWindows();
#endif
        }

        free_image(im);
        free_image(sized);
        if (filename) break;
    }
}

/*
void censor_detector(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename, int class, float thresh, int skip)
{
#ifdef OPENCV
    char *base = basecfg(cfgfile);
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    srand(2222222);
    CvCapture * cap;

    int w = 1280;
    int h = 720;

    if(filename){
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    if(w){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
    }
    if(h){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
    }

    if(!cap) error("Couldn't connect to webcam.\n");
    cvNamedWindow(base, CV_WINDOW_NORMAL); 
    cvResizeWindow(base, 512, 512);
    float fps = 0;
    int i;
    float nms = .45;

    while(1){
        image in = get_image_from_stream(cap);
        //image in_s = resize_image(in, net->w, net->h);
        image in_s = letterbox_image(in, net->w, net->h);
        layer l = net->layers[net->n-1];

        float *X = in_s.data;
        network_predict(net, X);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, in.w, in.h, thresh, 0, 0, 0, &nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

        for(i = 0; i < nboxes; ++i){
            if(dets[i].prob[class] > thresh){
                box b = dets[i].bbox;
                int left  = b.x-b.w/2.;
                int top   = b.y-b.h/2.;
                censor_image(in, left, top, b.w, b.h);
            }
        }
        show_image(in, base);
        cvWaitKey(10);
        free_detections(dets, nboxes);


        free_image(in_s);
        free_image(in);


        float curr = 0;
        fps = .9*fps + .1*curr;
        for(i = 0; i < skip; ++i){
            image in = get_image_from_stream(cap);
            free_image(in);
        }
    }
    #endif
}

void extract_detector(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename, int class, float thresh, int skip)
{
#ifdef OPENCV
    char *base = basecfg(cfgfile);
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    srand(2222222);
    CvCapture * cap;

    int w = 1280;
    int h = 720;

    if(filename){
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    if(w){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
    }
    if(h){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
    }

    if(!cap) error("Couldn't connect to webcam.\n");
    cvNamedWindow(base, CV_WINDOW_NORMAL); 
    cvResizeWindow(base, 512, 512);
    float fps = 0;
    int i;
    int count = 0;
    float nms = .45;

    while(1){
        image in = get_image_from_stream(cap);
        //image in_s = resize_image(in, net->w, net->h);
        image in_s = letterbox_image(in, net->w, net->h);
        layer l = net->layers[net->n-1];

        show_image(in, base);

        int nboxes = 0;
        float *X = in_s.data;
        network_predict(net, X);
        detection *dets = get_network_boxes(net, in.w, in.h, thresh, 0, 0, 1, &nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

        for(i = 0; i < nboxes; ++i){
            if(dets[i].prob[class] > thresh){
                box b = dets[i].bbox;
                int size = b.w*in.w > b.h*in.h ? b.w*in.w : b.h*in.h;
                int dx  = b.x*in.w-size/2.;
                int dy  = b.y*in.h-size/2.;
                image bim = crop_image(in, dx, dy, size, size);
                char buff[2048];
                sprintf(buff, "results/extract/%07d", count);
                ++count;
                save_image(bim, buff);
                free_image(bim);
            }
        }
        free_detections(dets, nboxes);


        free_image(in_s);
        free_image(in);


        float curr = 0;
        fps = .9*fps + .1*curr;
        for(i = 0; i < skip; ++i){
            image in = get_image_from_stream(cap);
            free_image(in);
        }
    }
    #endif
}
*/

/*
void network_detect(network *net, image im, float thresh, float hier_thresh, float nms, detection *dets)
{
    network_predict_image(net, im);
    layer l = net->layers[net->n-1];
    int nboxes = num_boxes(net);
    fill_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 0, dets);
    if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
}
*/

void run_detector(int argc, char **argv)
{
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .5);
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    int avg = find_int_arg(argc, argv, "-avg", 3);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");
    int fullscreen = find_arg(argc, argv, "-fullscreen");
    int width = find_int_arg(argc, argv, "-w", 0);
    int height = find_int_arg(argc, argv, "-h", 0);
    int fps = find_int_arg(argc, argv, "-fps", 0);
    //int class = find_int_arg(argc, argv, "-class", 0);

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    if(0==strcmp(argv[2], "test")) test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen);
    if(0==strcmp(argv[2], "folder")) folder_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen);
    else if(0==strcmp(argv[2], "train")) train_detector(datacfg, cfg, weights, gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "valid")) validate_detector(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "valid2")) validate_detector_flip(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "recall")) validate_detector_recall(datacfg, cfg, weights, thresh);
    else if(0==strcmp(argv[2], "demo")) {
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
        demo(cfg, weights, thresh, cam_index, filename, names, classes, frame_skip, prefix, avg, hier_thresh, width, height, fps, fullscreen);
    }
    //else if(0==strcmp(argv[2], "extract")) extract_detector(datacfg, cfg, weights, cam_index, filename, class, thresh, frame_skip);
    //else if(0==strcmp(argv[2], "censor")) censor_detector(datacfg, cfg, weights, cam_index, filename, class, thresh, frame_skip);
}
