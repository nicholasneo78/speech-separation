from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

model = separator.from_hparams(source="speechbrain/resepformer-wsj02mix", savedir='pretrained_models/resepformer-wsj02mix')

# for custom file, change path
est_sources = model.separate_file(path='/speech_separation/datasets/example_libri/mms_1_8000.wav') 

torchaudio.save("/speech_separation/datasets/example_libri/source1hat.wav", est_sources[:, :, 0].detach().cpu(), 8000)
torchaudio.save("/speech_separation/datasets/example_libri/source2hat.wav", est_sources[:, :, 1].detach().cpu(), 8000)

