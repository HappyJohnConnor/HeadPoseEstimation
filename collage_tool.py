# -*- coding: utf-8 -*-
from glob import glob
from wand.image import Image
import os, sys

def __saveImage( image, path ) :
    '''
    指定されたパスに画像を保存
    @param image
    @param path
    '''
    image.save( filename = path )

def mergeImages( src_filenames, part_width, part_height, row_max, col_max ) :
    '''
    入力ファイルパスの画像を張り付ける。足りない場合はループで回す。
    画像は全て同じサイズであることを前提としている。
    @param src_filenames 画像のファイルパス ( 複数 )
    @param part_width    画像一枚の幅
    @param part_height   画像一枚の高さ
    @param row_max       縦の画像枚数最大値
    @param col_max       横の画像枚数最大値
    '''
    merged_image = Image( width = ( part_width * col_max ), height = ( part_height * row_max ) )

    file_count = 0
    file_max   = len( src_filenames )
    for j in range( row_max ) :
        for i in range( col_max ) :
            image = Image( filename = src_filenames[file_count] )
            merged_image.composite( image, left = ( part_width * i ), top = ( part_width * j ) )
            file_count += 1

            if file_count >= file_max :
                file_count = 0

    return merged_image

if __name__ == '__main__' :

    argv = sys.argv
    argc = len( argv )
    
    """
    if argc < 6 :
        print("Usage: $ python {0} <src_dir> <dest_file> <size> <row_max> <col_max>").format( argv[0] ))
        quit()
    """
    src_dir       = argv[1]
    dest_filename = argv[2]
    size          = argv[3]
    row_max       = argv[4]
    col_max       = argv[5]
    
    # 入力ディレクトリ無ければエラー
    if not os.path.exists( src_dir ) :
        print("Source directory \"{0}\" is not found.".format( src_dir ))

    # 縦横枚数正しくなければエラー
    try :
        size    = int( size )
        row_max = int( row_max )
        col_max = int( col_max )
    except Exception as e :
        print( "Size parameter \"({0}, {1}, {2})\" is incorrect format./ {3}".format( size, row_max, col_max, e ))
        quit()
    
    # 実行
    image = mergeImages( glob( "{0}/*".format( src_dir ) ), size, size, row_max, col_max )
    __saveImage( image, dest_filename )