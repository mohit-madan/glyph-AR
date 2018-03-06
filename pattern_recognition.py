import cv2

# takes a glyph pattern array and extends by a black border. Increases in size by 2 in both dimensions but still stored as array
def return_with_black_pattern(glyph_pattern,black_pattern,glyph_size):
    grid_size = glyph_size  + 2
    #black_pattern_in = [0 for x in range(grid_size*grid_size)]
    black_pattern_in = black_pattern.copy()
    for i in range(glyph_size):
        position_full = (i+1)*grid_size + 1
        position_glyph = i*glyph_size
        black_pattern_in[position_full : (position_full+glyph_size)] = glyph_pattern[position_glyph:(position_glyph+glyph_size)]
    return black_pattern_in

# rotates elements in a matrix 90 degrees clockwise around center
# Since arrays are used index is transformed correspondingly
# https://stackoverflow.com/questions/42519/how-do-you-rotate-a-two-dimensional-array
def rotate_glyph(glyph, n):
    res = [0 for x in range(n*n)]
    for i in range(n):
        for j in range(n):
            res[i*n+j] = glyph[(n - j - 1)*n+i]
    return res

def generate_black_pattern(glyph_size):
    result = [0 for x in range((glyph_size+2)*(glyph_size+2))]
    for i in range(glyph_size):
        position = (i+1)*(glyph_size+2) + 1
        result[position : position+glyph_size] = [1 for x in range(glyph_size)]

    return result

def pattern_recognition(image):
    # If no glyphs provided by input use standard glyphs todo
    input_glyph = image
    GLYPH_SIZE = 3
    NUM_GLYPHS = 2
    glyphs = [[0 for x in range(GLYPH_SIZE * GLYPH_SIZE)] for x in range (NUM_GLYPHS)]
    glyphs[0] = [0, 1, 0, 1, 0, 0, 0, 1, 1]
    glyphs[1] = [1, 0, 0, 0, 1, 0, 1, 0, 1]

    RESIZE_SIZE = 100
    BLACK_THRESHOLD = 100
    BLACK_PERCENT = 0.75

    resized_input = cv2.resize(input_glyph, (100,100))
    grid_size = GLYPH_SIZE + 2
    black_pattern = generate_black_pattern(GLYPH_SIZE)
    # add black border to glyphs and add rotated version to database
    # same glyphs will have same idx\4 (integer division)
    glyphs_black = [[0 for x in range(grid_size*grid_size)] for x in range (NUM_GLYPHS*4)]
    for i in range(len(glyphs)):
        glyphs_black[i*4] = return_with_black_pattern(glyphs[i],black_pattern,GLYPH_SIZE)
        for j in range(3):
            glyphs[i] = rotate_glyph(glyphs[i],GLYPH_SIZE)
            glyphs_black[i*4 +j+1] = return_with_black_pattern(glyphs[i],black_pattern,GLYPH_SIZE)


    # create map for input glyph
    glyph_pattern = [0 for x in range ((grid_size) * (grid_size))] #include border in pattern, has to be 0
    for grid_x in range(grid_size):
        for grid_y in range(grid_size):
            # check whether square in (x,y) ist black or white
            # iterate over each element in grid
            num_pixels = (int) (RESIZE_SIZE/grid_size) #height and width of each grid element
            sum_black = 0
            for x in range(num_pixels):
                for y in range(num_pixels):
                    im_x = grid_x * num_pixels + x
                    im_y = grid_y * num_pixels + y

                    if (resized_input[im_x][im_y] < BLACK_THRESHOLD):
                        sum_black = sum_black + 1
            # enough black cells?
            if (sum_black/(num_pixels*num_pixels) < BLACK_PERCENT):
                glyph_pattern[grid_x*grid_size +grid_y] = 1

    # compare result with our glyph database
    # return found and idx of glyph
    glyph_found = False
    for i in range(len(glyphs_black)):
        if (glyphs_black[i] == glyph_pattern):
            glyph_found = True
            glyph_idx = i//4
            print('Found matching glyph!')
            print(glyph_idx)
            break

    if (glyph_found):
        return glyph_idx
    else:
        return None
