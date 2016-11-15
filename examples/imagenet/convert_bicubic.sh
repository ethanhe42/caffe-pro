for name in ~/heyihui-local/bicubicval/*.JPEG; do
    convert -interpolate bicubic -resize 256x256\! $name $name && echo $name
done
