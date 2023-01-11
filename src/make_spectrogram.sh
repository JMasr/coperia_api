#!/bin/bash
# aspec.sh
# get spectrograms of audio streams
#
# usage: aspec.sh a.mp3 b.m4a c.mp4 d.mkv ....
#
# dependencies: sox, ffmpeg
# license: public domain, warranty: none
# version: 2019-05-17 by milahu

ff_args="" # ffmpeg arguments
sx_args="" # sox arguments

ff_args+=" -loglevel error"

ff_astream=0 # only use first audio stream
ff_args+=" -map 0:a:${ff_astream}?"

ff_args+=" -ac 1" # use only one audio channel
sx_args+=" channels 1"

sx_args+=" gain -n -3" # normalize volume to -3dB

# set sampling rate
# only analyze frequencies below f_max = rate / 2
# also normalize spectrogram height to f_max
#sx_args+=" rate 6k"  # only show f <  3kHz "where the human auditory system is most sensitive"
sx_args+=" rate 48k" # only show f < 24kHz

# use wav as temporary format, if sox cant read file
ff_args+=" -c:a pcm_s16le -f wav"
sx_type="wav"

# process files from "argv"
for i in "$@"
do
    echo "$i"
    o="${i%.*}.png" # output file
    t=$(basename "$i") # title above spectrogram
    c="spectrogram by SoX, the Sound eXchange tool" # comment below spectrogram

    # try to read original format
    echo analyze
    sox "$i" -n \
        $sx_args \
        spectrogram -o "$o" -c "$c" -t "$t" \
        2>&1 | grep -v "no handler for detected file type"

    if (( ${PIPESTATUS[0]} != 0 ))
    then
        # sox failed. convert audio and retry
        echo convert

        # get duration of stream or container
        # spectrogram filter has no "ignore length" option
        # and without a "duration prediction" will only read 8 seconds
        d=$(ffprobe "$i" -v error -of compact=s=_ \
            -select_streams "0:a:${ff_astream}?" \
            -show_entries stream=duration:format=duration \
            | sort | grep -v =N/A \
            | tail -n 1 | cut -d= -f2)
        # 'tail -n 1' --> prefer stream duration
        # 'head -n 1' --> prefer container duration

        if [[ -z "$d" ]]
        then
            echo -e "skip. duration not found FIXME\n"
            continue
        fi

        # bash "process substitution" magic
        sox \
            --type "$sx_type" \
            --ignore-length \
            <( ffmpeg -i "$i" $ff_args - ) \
            --null \
            $sx_args \
            spectrogram -d "$d" -o "$o" -c "$c" -t "$t"
    fi

    echo -e "done\n$o\n"
done