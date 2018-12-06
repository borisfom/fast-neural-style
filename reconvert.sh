GPU=$1
INPUT=$2

export CUDA_VISIBLE_DEVICES=${GPU}

mkdir -p new-onnx/rel8
mkdir -p new-engines

mkdir -p new-styles
mkdir -p saved-models
# for the first time : 
# (cd saved-models && ../download_styling_models.sh)
# Download models from Google Drive to new-styles


OLDSTYLES="mosaic candy udnie starry-night"
for m in ${OLDSTYLES}; do python neural_style/neural_style.py eval  --content-image ${INPUT} --output-image new-onnx/${INPUT}-$m.jpg --cuda 1 --export_onnx new-onnx/rel8/$m.onnx --model saved-models/$m.pth ; done

NEWSTYLES="kandinsky picasso sharaku-196 klimt sharaku-197 sharaku monet"

for m in ${NEWSTYLES}; do python neural_style/neural_style.py eval --content-image ${INPUT} --output-image new-onnx/bayou-${INPUT}.jpg --cuda 1 --export_onnx new-onnx/rel8/$m.onnx --model new-styles/$m.pth ; done

for f in ${OLDSTYLES} ${NEWSTYLES}; do onnx2trt -o ./new-engines/$f.engine -b 1 -w 8000000000 -d 16 -l -v ./new-onnx/rel8/$f.onnx ; done

cd new-engines

mv picasso.engine  picasso3.engine 
mv mosaic.engine mosaic-fp16.engine 
mv starry-night.engine starry-night-fp16.engine 
mv udnie.engine udnie-fp16.engine 
mv candy.engine candy-fp16.engine 
mv klimt.engine kiss.engine 
mv sharaku-197.engine ebizo.engine 
mv sharaku-196.engine matsumoto.engine

