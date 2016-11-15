for name in $1/*.JPEG; do
        convert -resize 256x256\! $name $name
done
