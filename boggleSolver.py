import numpy as np
import cv2
import os


class boggleSolver:
    def __init__(self, model, showImages=True, verbose = True, distPercent=.75):
        self.im            = []
        self.distPercent   = distPercent
        self.showImages    = showImages
        self.model         = model
        self.verbose       = verbose

        if self.verbose:
            print("Initialized boggle solver")
        
    # Process input image and find 16 individual images of
    # each letter, classify each letter, return letters and output image
    def process_image(self, im):
        # Input image
        im = cv2.resize(im,(480,640))
        imBlur  = im.copy()
        im_og   = im.copy()
        self.im = im
        if self.showImages:
            imshow(im,"og im")
            imshow(imBlur,"blur im")

        # Blur 
        imBlur = cv2.GaussianBlur(imBlur,(5,5),0)

        # Threshold non-white values
        mask_rgb = cv2.inRange(imBlur, (100,100,100), (255,255,255))
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_GRAY2BGR)
        imBlur = imBlur & mask_rgb
        if self.showImages:
            imshow(imBlur,'Thresholded Im')

        # Convert to grayscale if it's color
        try:
            if im.shape[2] > 1:
                im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
                imBlur = cv2.cvtColor(imBlur,cv2.COLOR_BGR2GRAY)
        except:
            pass

        # Convert to binary
        im     = cv2.threshold(im,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        imBlur = cv2.threshold(imBlur,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #imBlur = cv2.bitwise_not(imBlur)
        #print(self.showImages)
        if self.showImages:
            imshow(im, 'binary image')
            imshow(imBlur, 'binary blur')
        if self.verbose:
            print("Converted image to binary")
        im = imBlur

        # Add Padding
        # m,n = im.shape
        # im_left  = np.zeros((m+1,1))                  # padding to the left
        # im_right = np.bmat([[np.zeros((1,n))],[im]])  # padding on top
        # im_pad   = np.bmat([[im_left,im_right]])
        # im_pad   = im_pad.astype(np.uint8)
        # if self.verbose:
        #     print("Added padding")
        # if self.showImages:
        #     imshow(im_pad,'Padded Im')
        im_pad = im
        
        # Run Contour for boggle tiles
        contours, hierarchy = cv2.findContours(im_pad, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        center_points = []
        good_contours = []
        for con in contours:
            area = cv2.contourArea(con)
            perimeter = cv2.arcLength(con, True)
            if perimeter == 0:
                continue
            circularity = 4*np.pi*(area/(perimeter*perimeter))
            if area>3000 and (0.6 < circularity < 1.2):
                x,y,w,h = cv2.boundingRect(con)
                center_points.append([x+w/2, y+h/2])
                good_contours.append(con)
        center_points = np.asarray(center_points)
        if self.verbose:
            print("Num contours found: " + str(len(good_contours)))

        # Get center point for each letter
        #keypoints = self.find_blobs(im_pad) # blob detector
        #center_points = []
        #for i in range(len(keypoints)):
        #    center_points.append(keypoints[i].pt)
        #center_points = np.asarray(center_points)
        if self.verbose:
            print("Found center points for each letter")

        # If we have 16 contours just use contour information to
        # grab letters, if not use the center_points from the contour finder
        # and run the grid point check
        if len(good_contours) == 16:

            if self.verbose:
                print("Using contours")

            # Order points
            y_weight = 5 
            sorted_inds = np.argsort((1+(center_points[:,1]/max(center_points[:,1])))*y_weight \
                + center_points[:,0]/max(center_points[:,0]))
            sorted_good_contours = [good_contours[i] for i in sorted_inds]
            good_contours = sorted_good_contours
            center_points = center_points[sorted_inds]

             # Separate each individual letter
            letters = []
            boxedIm = self.im.copy()
            for con in good_contours:
                x,y,w,h = cv2.boundingRect(con)
                croppedIm = self.im[int(y):int(y+h), int(x):int(x+w)]
                crop_px = 10
                croppedIm = croppedIm[crop_px:croppedIm.shape[0]-crop_px, crop_px:croppedIm.shape[1]-crop_px]
                letters.append(croppedIm)
                #imshow(croppedIm,'a')
                cv2.rectangle(boxedIm,(int(x),int(y)),(int(x+w),int(y+h)),(0,255,0),2)
                cv2.drawContours(boxedIm,con,-1,(0,0,255),2)
            if self.showImages:
                imshow(boxedIm,'boxed letters')

        else:

            if self.verbose:
                print("Using center points")

            # Get distance between center points
            dist = self.get_dist_between_letters(center_points)
            if self.verbose:
                print("Got avg distance between letters")

            # Get rid of outlier points and fill in letter points we may have missed
            temp = self.grid_point_check(center_points,dist,20,20,self.im)
            center_points = temp
            if self.verbose:
                print("Grid point check done")
    
            # Order coordinates as as [1;2;....;16] so that the board looks like:
            # ---------------        
            # | 1  2  3  4  |
            # | 5  6  7  8  |
            # | 9  10 11 12 |
            # | 13 14 15 16 |
            # ---------------
            y_weight = 5 
            center_points_ordered = center_points[np.argsort((1+(center_points[:,1]/max(center_points[:,1])))*y_weight \
                + center_points[:,0]/max(center_points[:,0]))] 
            center_points = center_points_ordered

            # Separate each individual letter
            letters = []
            boxLen  = self.distPercent * dist
            boxedIm = self.im.copy()
            for i in range(len(center_points)):
                croppedIm = self.im[int(center_points[i][1]-boxLen/2.0):int(center_points[i][1]+boxLen/2.0), int(center_points[i][0]-boxLen/2.0):int(center_points[i][0]+boxLen/2.0)]
                letters.append(croppedIm)
                #if self.showImages:
                if i < len(good_contours):
                    con = good_contours[i]
                    cv2.drawContours(boxedIm,con,-1,(0,0,255),2)
                cv2.rectangle(boxedIm,(int(center_points[i][0]-boxLen/2.0),int(center_points[i][1]-boxLen/2.0)),(int(center_points[i][0]+boxLen/2.0),int(center_points[i][1]+boxLen/2.0)),(0,255,0),2)
            if self.showImages:
                imshow(boxedIm,'boxed letters')

        # Show each cropped letter
        #if self.showImages:
        #    for i in range(len(letters)):
        #        imshow(letters[i],"letter")
        #        #print(center_points_ordered[i,:])
 
        #TODO: Put this into a function
        # Iterate through letters and classify them
        letter_vals = []
        for i in range(len(letters)):
            thisLetterVals = [] # Holds classified letter for each rotated version of this letter
            thisLetter,confidence = self.predict_letter(letters[i])
            thisLetterVals.append(thisLetter[0])
            letterUsed = cv2.resize(letters[i],(40,40))
            #imshow(letterUsed,thisLetter+': '+str(confidence))
            (hh,ww) = letters[i].shape[:2]
            cent = (ww/2,hh/2)
            ang = 90
            letter_guess = np.array([thisLetter])            # Array of predicted letters for all four rotations of current letter
            letter_guess_confidence = np.array([confidence]) # Array of predicted letter confidence corresponding to letter_guess
            for r in range(3):
                M = cv2.getRotationMatrix2D(cent,ang,1)           # Get rotation matrix
                rot_letter = cv2.warpAffine(letters[i],M,(hh,ww)) # Rotate letter image
                thisLetterRot,confidence2 = self.predict_letter(rot_letter) # Get letter prediction/confidence
                ang += 90

                # If we've never predicted this letter add it to our array
                # If we have, add current confidence value to the exisitng corresponding letter confidence
                thisLetterVals.append(thisLetterRot[0])
                if thisLetterRot[0] in letter_guess:
                    letter_guess_confidence[np.nonzero(letter_guess==thisLetterRot[0])] += confidence2
                else:
                    letter_guess= np.append(letter_guess,thisLetterRot[0])
                    letter_guess_confidence = np.append(letter_guess_confidence,confidence2)

            # After adding up confidences for each rotated version of this letter,
            # Choose the letter with the highest cumulative confidence as our prediction
            thisLetter = letter_guess[np.argmax(letter_guess_confidence)]
            if thisLetter == "Q":
                thisLetter = "Qu"
            letter_vals.append(thisLetter)
        
        if True:#self.showImages:
            # Create output image of the board we saw, 'letters' is in row-major order
            output_im = np.zeros((im.shape))
            for i in range(len(letters)):
                cv2.putText(output_im, letter_vals[i], (int(center_points[i][0]),int(center_points[i][1])),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
        
            # Put input image next to output image
            output_im_3channel = np.zeros_like(im_og)
            output_im_3channel[:,:,0] = output_im
            output_im_3channel[:,:,1] = output_im
            output_im_3channel[:,:,2] = output_im
            #input_output_im = np.hstack((im_og,output_im_3channel))
            input_output_im = np.hstack((boxedIm,output_im_3channel))
            #cv2.imshow('Input vs Output',input_output_im)
            #cv2.waitKey()
            #cv2.destroyAllWindows()

        # Format letter_vals into list of rows for solver
        l = np.reshape(letter_vals,[4,4])
        letters_out = [l[0,:].tolist(), l[1,:].tolist(), l[2,:].tolist(), l[3,:].tolist()]

        return input_output_im, letters_out

    #TODO: two coordinates that are super close into one point
    def merge_close_points(self,points):
        output_points = []
        for i in range(len(points)):
            pt_min = np.argmin()

    # Find each letter on the boggle board. Return points of center of 
    # each letter
    def find_blobs(self,im_pad):
       params = cv2.SimpleBlobDetector_Params()
       params.filterByInertia     = False # Want low inertia?
       params.filterByConvexity   = True  # This works really well on the boggle tiles
       params.filterByCircularity = False
       params.filterByColor       = False # why doesn't this work
       params.filterByArea        = True  # Gets rid of small noise
       params.minArea = 600
       params.blobColor = 255
       params.minConvexity = 0.87
       detector = cv2.SimpleBlobDetector_create(params)
       keypoints = detector.detect(im_pad)
       if self.verbose:
           print("Num keypoints found: " + str(len(keypoints)))

       if len(keypoints) < 10:
            params = cv2.SimpleBlobDetector_Params()
            params.filterByInertia     = False # Want low inertia?
            params.filterByConvexity   = True  # This works really well on the boggle tiles
            params.filterByCircularity = False
            params.filterByColor       = False # why doesn't this work
            params.filterByArea        = True
            params.minArea = 200
            params.minConvexity = 0.3
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(im_pad)
            if self.verbose:
                print("New num keypoints found: " + str(len(keypoints)))

       if self.showImages:
            im_with_keypoints = cv2.drawKeypoints(self.im, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            imshow(im_with_keypoints,'im with keypoints')

        # keypoints now has center point for each letter
       return keypoints


    def get_dist_between_letters(self,center_points):

        # Get average/median distance between 'grid' points.
        # Have center point of each letter. Get distance of letter vertical and horizontal,
        # get average of these distances, use that avgDist to draw a box around the center of each
        # letter. Box length dependent on avgDist

        # Min dist from each center points to any other center point.
        min_dist = []
        for i in range(len(center_points)):
            pt_diff = np.sqrt([(center_points[i,0] - center_points[:,0])**2 + (center_points[i,1]-center_points[:,1])**2]) # diff between this center point and all others
            pt_diff[0,i] = 1000 # So that min value to itself won't be 0
            min_dist.append(np.min(pt_diff)) # Get the minimum distance from this center point to another centor point

        # Have min_dist, a list of the minimum distance from each center point to any other center points
        # Take the average maybe of this then use a percentage of that to draw a box around each center point
        # with length = percentage * avgDist
        dist = np.median(np.asarray(min_dist))

        return dist

    def predict_letter(self,letter):

        #labels = ['S270', 'D180', 'A90', 'D90', 'H90', 'S90', 'T270', 'T90', 'R', 'V270', 'V90', 'T', 'V', 'K', 'E180', 'V180', 'D270', 'E90', 'E', 'E270', 'R90', 'A180', 'A270', 'K270', 'Y270', 'O', 'Y', 'Y90', 'D', 'K180', 'H', 'T180', 'K90', 'R180', 'A', 'Y180', 'R270', 'S']
        labels = ['S270', 'G270', 'Q90', 'D180', 'L', 'W', 'Z180', 'C90', 'A90', 'L180', 'J180', 'X', 'B180', 'C', 'Z', 'U270', 'D90', 'H90', 'S90', 'G90', 'B90', 'T270', 'P180', 'T90', 'F90', 'N90', 'L270', 'P', 'R', 'V270', 'V90', 'T', 'V', 'W180', 'K', 'W270', 'M90', 'X90', 'E180', 'V180', 'D270', 'E90', 'M270', 'E', 'E270', 'F', 'U90', 'R90', 'A180', 'P90', 'A270', 'K270', 'B270', 'M180', 'Y270', 'L90', 'W90', 'Q', 'M', 'F180', 'Q270', 'Q180', 'O', 'J90', 'C180', 'Y', 'U', 'B', 'Y90', 'D', 'U180', 'K180', 'J', 'H', 'N', 'C270', 'I90', 'Z270', 'T180', 'K90', 'R180', 'F270', 'A', 'P270', 'G', 'Z90', 'I', 'Y180', 'R270', 'S', 'J270', 'G180']
        sz     = 40 # Size used in training
        #b,g,r  = cv2.split(letter)
        #letter = cv2.merge((r,g,b))         # Make it rgb (was bgr)
        letter = cv2.cvtColor(letter,cv2.COLOR_BGR2GRAY) # color to grayscale
        letter = cv2.threshold(letter,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] # grayscale to binary
        letter = cv2.resize(letter,(sz,sz)) # resize image size
        letter = letter.astype(float)       # change type to float
        letter = letter/255.0               # normalize

        # Push letter image through the network
        im = letter.reshape(1,40,40,1)                        # add extra dimension for model input
        confidence = np.max(self.model.predict(im,verbose=0)) # get confidence as softmax output
        thisLetter = labels[self.model.predict_classes(im)[0]]
        #print("Confidence: " , confidence)
        #print("Letter: " , thisLetter)
        #print("\n")
        thisLetter = thisLetter[0] # Get letter (ex: R180 -> R)
        confidenceIndex = np.argsort(self.model.predict(im,verbose=0)[0])[::-1][0:3]
        
        #print("----------------------------")
        #for ind in confidenceIndex:
        #     print(labels[ind],":", self.model.predict(im,verbose=0)[0][ind])
        #print("----------------------------")

        return thisLetter, confidence


        # Rotate image and take max confidence of the 4 options. Hopefully highest confidence will be when letter is upright
        # highest_confidence = -1
        # for _ in range(4):
        #     im = letter.reshape(1,40,40,1)                        # add extra dimension for model input
        #     confidence = np.max(self.model.predict(im,verbose=0)) # get confidence as softmax output
        #     thisLetter = labels[self.model.predict_classes(im)[0]]
        #     print('thisLetter: ', thisLetter,', confidence: ',confidence)

        #      # If confidence for this rotation higher than current highest, set new highest and letter
        #      print('len(squares): ',len(squares))
        #      print('len(squares): ',len(squares))
        #      if confidence > highest_confidence:
        #          thisLetterHighest = labels[self.model.predict_classes(im)[0]]
        #          highest_confidence = confidence

        #      # Rotate letter image 90 degrees
        #      Rot    = cv2.getRotationMatrix2D((sz/2,sz/2),90,1)
        #      letter = cv2.warpAffine(letter,Rot,(sz,sz))
        #      print('\n')
        #return thisLetterHighest,highest_confidence


    # Take in measured letter points and return points that are where letters are
    # Filters out false detections and fills in missed detections
    # Slide window over x and y coords of image to get top 4 bin locations
    # INPUTS:
    #        center_points --- detected letter points
    #        dist          --- pixel distance between letters on board
    #        range         --- range of sliding bin
    #        step_length   --- length of step to move over x and y points
    # OUTPUTS:
    #        grid_points   --- letter locations on board
    def grid_point_check(self,center_points, dist, range, step_length, im):
        # Slide over x axis
        x_bins = []
        x_bins_points = []
        bin_center = int(range / 2)
        #TODO: do a quick check that with this image size and range and step length, we will have at least 4 bins
        while (bin_center < im.shape[1]):
            points_in_this_bin = []
            bin_left  = bin_center - (range/2) # left bound of bin
            bin_right = bin_center + (range/2) # right bound of bin
            num_points = 0                     # track num of points in this bin
            for point in center_points:
                if ((point[0]>=bin_left) and (point[0]<=bin_right)):
                    num_points += 1
                    points_in_this_bin.append([point])
            x_bins.append([bin_center, num_points])
            x_bins_points.append

            # Slide over bin 
            bin_center += step_length

        x_bins = np.asarray(x_bins) # into numpy array
        # Sort by 2nd column, num_points, puts least num points at top so use np.flip() to flip it
        x_bins_ordered = np.flip(x_bins[x_bins[:,1].argsort()])
        top4_x = np.array([x_bins_ordered[0],x_bins_ordered[1],x_bins_ordered[2],x_bins_ordered[3]])

        # Slide over y axis
        y_bins = []
        bin_center = int(range / 2)
        # TODO: do a quick check that with this image size and range and step length, we will have at least 4 bins
        while (bin_center < im.shape[0]):
            bin_left  = bin_center - (range/2) # left bound of bin
            bin_right = bin_center + (range/2) # right bound of bin
            num_points = 0                     # track num of points in this bin
            for point in center_points:
                if ((point[1]>=bin_left) and (point[1]<=bin_right)):
                    num_points += 1
            y_bins.append([bin_center, num_points])

            # Slide over bin 
            bin_center += step_length

        y_bins = np.asarray(y_bins) # into numpy array
        # Sort by 2nd column, num_points, puts least num points at top so use np.flip() to flip it
        y_bins_ordered = np.flip(y_bins[y_bins[:,1].argsort()])
        top4_y = np.array([y_bins_ordered[0],y_bins_ordered[1],y_bins_ordered[2],y_bins_ordered[3]])

        # Plot top
        if self.showImages: 
            top4Im = im.copy()
            for x in top4_x:
                for y in top4_y:
                    cv2.rectangle(top4Im,(int(x[1]-dist/2.0),int(y[1]-dist/2.0)),(int(x[1]+dist/2.0),int(y[1]+dist/2.0)),(0,255,0),2)
                    #print('plotting x,y: ',x[1],',',y[1])
            imshow(top4Im,'top 4')

        good_points = []
        top_4_points_ind_used = []
        for pointInd,point in enumerate(center_points):
          #  print('-------------------------------')
            curr_dist = dist
            xIndUsed = -1
            yIndUsed = -1
            go_to_next_point = 0
            #print('----------------------------------------')
            for yind,y in enumerate(top4_y):
                for xind,x in enumerate(top4_x):
                    #print("\npoint\n",point)
                    #print("good_points\n",good_points)
                    #if len(good_points) > 0:
                    #    print("good_points[0]: ",good_points[0])
                    #print("point not in good_points: ",point not in good_points)
                    p_dist = ((point[0]-x[1])**2 + (point[1]-y[1])**2)**.5
                    #print('p_dist: ',p_dist)
                    if (p_dist < curr_dist):# and point not in good_points):
                        curr_dist = p_dist
                        xIndUsed = xind
                        yIndUsed = yind
                        #print('xind: ', xind, ' yind: ', yind, ' p_dist = ', p_dist)
                        #good_points.append(list(point))
                        #top_4_points_ind_used.append([xind,yind])
                        #print('this point: ', point)
                        #go_to_next_point = 1
                        #break
                #if go_to_next_point == 1:
                #    break
            if curr_dist < dist:
                good_points.append(list(point))
                top_4_points_ind_used.append([xIndUsed,yIndUsed])
        #print('len of good points: ',len(good_points))
        #print('top_4_points_used: ', top_4_points_ind_used)
        #print('len top 4 points used: ', len(top_4_points_ind_used))
        #print('')

        # If we have less than 16 points, fill in the missing points
        # from top4_x and top4_y points that didn't match anything in center_points
        if len(good_points) < 16:
            wanted_inds = [[0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3],[3,0],[3,1],[3,2],[3,3]]
            for this_top4_ind in top_4_points_ind_used:
                if this_top4_ind in wanted_inds:
                    wanted_inds.remove(this_top4_ind)

            # Add the missing points
            for w_ind in wanted_inds:
                wanted_x_ind = w_ind[0]
                wanted_y_ind = w_ind[1]
                wanted_coord = [top4_x[wanted_x_ind][1], top4_y[wanted_y_ind][1]]
                good_points.append(wanted_coord)
        #print('Unfound inds: ', wanted_inds)
        #print('Final good poitns: ', good_points)
        #print(top4_x[0][1])

        # If more than 16 points just use the top4 coords
        #TODO: More filters for merging/removing extra points
        bin_points = []
        if len(good_points) > 16:
            for x in top4_x:
                for y in top4_y:
                    bin_points.append([x[1], y[1]])
            good_points = bin_points            

        return np.asarray(good_points)
       

# Not being used right now, find squares in image
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )
# Not being used right now
def find_squares(img,blurVal,dilateVal,minArea,maxArea):
    img = cv2.GaussianBlur(img, (blurVal, blurVal), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in xrange(0, 255, dilateVal):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            bin, contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            #imshow(bin,'f')
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > minArea and cv2.contourArea(cnt) < maxArea and cv2.isContourConvex(cnt):
                     cnt = cnt.reshape(-1, 2)
                     max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                     if max_cos < 0.1:
                         squares.append(cnt)
                    #squares.append(cnt)
    return squares

# Display an image
def imshow(im,str):
    cv2.imshow(str,im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
