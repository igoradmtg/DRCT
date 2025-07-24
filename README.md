# DRCT: Спасение сверхразрешения изображений от информационного узкого места

### ✨✨ [Устная презентация CVPR NTIRE]

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/drct-saving-image-super-resolution-away-from/image-super-resolution-on-set5-4x-upscaling)](https://paperswithcode.com/sota/image-super-resolution-on-set5-4x-upscaling?p=drct-saving-image-super-resolution-away-from)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/drct-saving-image-super-resolution-away-from/image-super-resolution-on-urban100-4x)](https://paperswithcode.com/sota/image-super-resolution-on-urban100-4x?p=drct-saving-image-super-resolution-away-from)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/drct-saving-image-super-resolution-away-from/image-super-resolution-on-set14-4x-upscaling)](https://paperswithcode.com/sota/image-super-resolution-on-set14-4x-upscaling?p=drct-saving-image-super-resolution-away-from)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/drct-saving-image-super-resolution-away-from/image-super-resolution-on-manga109-4x)](https://paperswithcode.com/sota/image-super-resolution-on-manga109-4x?p=drct-saving-image-super-resolution-away-from)

[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/igoradmtg/DRCT/issues)

---

## [[Ссылка на статью]](https://arxiv.org/abs/2404.00722) [[Страница проекта]](https://allproj002.github.io/drct.github.io/) [[Постер]](https://drive.google.com/file/d/1zR9wSwqCryLeKVkJfTuoQILKiQdf_Vdz/view?usp=sharing) [[Модельный зоопарк]](https://drive.google.com/drive/folders/1QJHdSfo-0eFNb96i8qzMJAPw31u9qZ7U?usp=sharing) [[Визуальные результаты]](https://drive.google.com/drive/folders/15raaESdkHD-7cHWBVDzDitTH8_h5_0uE?usp=sharing) [[Слайды]](https://docs.google.com/presentation/d/1MxPPtgQZ61GFSr3YfGOm9scm23bbbXRj/edit?usp=sharing&ouid=105932000013245886245&rtpof=true&sd=true) [[Видео]](https://drive.google.com/file/d/17dB47E8I2ME-shhxAWDlQCyCuJRn79d_/view?usp=sharing)

[Chih-Chung Hsu](https://cchsu.info/), [Chia-Ming Lee](https://igoradmtg.github.io/), [Yi-Shiuan Chou](https://nelly0421.github.io/)

Advanced Computer Vision LAB, National Cheng Kung University

---

## Обзор (SwinIR с плотными связями)

---

### Предпосылки и мотивация

В методах сверхразрешения (SR) на основе CNN плотные связи широко считаются эффективным способом сохранения информации и повышения производительности (представлено RDN / RRDB в ESRGAN... и т. д.).

Однако методы на основе SwinIR, такие как HAT, CAT, DAT и т. д., обычно используют блок внимания к каналам (Channel Attention Block) или разрабатывают новые и сложные механизмы внимания сдвига окна (Shift-Window Attention Mechanism) для улучшения производительности SR. Эти работы игнорируют **информационное узкое место**, из-за которого поток информации будет теряться глубоко в сети.

### Основной вклад

Наша работа **просто добавляет плотные связи** в SwinIR для улучшения производительности и вновь **подчеркивает важность плотных связей** в методах SR на основе Swin-IR. Добавление плотных связей в глубокую экстракцию признаков может стабилизировать поток информации, тем самым повышая производительность и сохраняя легковесность дизайна (по сравнению с современными методами, такими как HAT).

---

<img src=".\figures\overview.png" width="500"/>

<img src=".\figures\drct_fix.gif" width="600"/>

<img src=".\figures\4.png" width="400"/>

**Результаты бенчмарка для SRx4 без предварительного обучения x2. Multi-Adds рассчитаны для входа 64x64.**
| Модель | Параметры | Multi-Adds | Forward | FLOPs | Set5 | Set14 | BSD100 | Urban100 | Manga109 | Журнал обучения |
|:-----------:|:---------:|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [HAT](https://github.com/XPixelGroup/HAT) | 20.77M | 11.22G | 2053M | 42.18G | 33.04 | 29.23 | 28.00 | 27.97 | 32.48 | - |
| [DRCT](https://drive.google.com/file/d/1jw2UWAersWZecPq-c_g5RM3mDOoc_cbd/view?usp=sharing) | 14.13M | 5.92G | 1857M | 7.92G | 33.11 | 29.35 | 28.18 | 28.06 | 32.59 | - |
| [HAT-L](https://github.com/XPixelGroup/HAT) | 40.84M | 76.69G | 5165M | 79.60G | 33.30 | 29.47 | 28.09 | 28.60 | 33.09 | - |
| [DRCT-L](https://drive.google.com/file/d/1bVxvA6QFbne2se0CQJ-jyHFy94UOi3h5/view?usp=sharing) | 27.58M | 9.20G | 4278M | 11.07G | 33.37 | 29.54 | 28.16 | 28.70 | 33.14 | - |
| [DRCT-XL (предварительно обучено на ImageNet)](https://drive.google.com/file/d/1uLGwmSko9uF82X4OPOMw3xfM3stlnYZ-/view?usp=sharing) | - | - | - | - | 32.97 / 0.91 | 29.08 / 0.80 | - | - | - | [журнал](https://drive.google.com/file/d/1kl2r9TbQ8TR-sOdzvCcOZ9eqNsmIldGH/view?usp=drive_link) |

**Real DRCT GAN SRx4. (Обновлено)**

| Модель | Данные обучения | Контрольная точка | Журнал |
|:-----------:|:---------:|:-------:|:--------:|
| [Real-DRCT-GAN_MSE_Model](https://drive.google.com/drive/folders/1emyaw6aQvhFgFC_RjK1Qo9c1sTRr-avk?usp=sharing) | [DF2K + OST300](https://www.kaggle.com/datasets/thaihoa1476050/df2k-ost/code) | [Контрольная точка](https://drive.google.com/drive/folders/1emyaw6aQvhFgFC_RjK1Qo9c1sTRr-avk?usp=sharing) | [Журнал](https://drive.google.com/file/d/1kl2r9TbQ8TR-sOdzvCcOZ9eqNsmIldGH/view?usp=drive_link) |
| [Real-DRCT-GAN_Finetuned from MSE](https://drive.google.com/drive/folders/1emyaw6aQvhFgFC_RjK1Qo9c1sTRr-avk?usp=sharing) | [DF2K + OST300](https://www.kaggle.com/datasets/thaihoa1476050/df2k-ost/code) | [Контрольная точка](https://drive.google.com/drive/folders/1emyaw6aQvhFgFC_RjK1Qo9c1sTRr-avk?usp=sharing) | [Журнал](https://drive.google.com/file/d/15aBV-FFi7I4esUb1vzRmrjMccc5cEEY4/view?usp=drive_link) |

---

## Сравнение Real DRCT GAN (Спасибо [Phhofm](https://github.com/Phhofm)!)

Изображения ниже демонстрируют возможности улучшения модели 4xRealWebPhoto_v4_drct-l по сравнению со стандартным масштабированием 4x по методу ближайшего соседа:

Продемонстрировать:
[Slow.pic](https://slow.pics/s/VOKVChT9) ссылка для интерактивного сравнения со слайдером

<img src=".\figures\real-drct.png" width="1000"/>

---

## Обновления

- ✅ 31.03.2024: Выпущенна первая версия статьи на Arxiv.
- ✅ 14.04.2024: DRCT принят NTIRE 2024, CVPR.
- ✅ 02.06.2024: Выпущенна предварительно обученная DRCT-L.
- ❌ 02.06.2024: Выпущенна MambaDRCT. [MODEL.PY](https://drive.google.com/file/d/1di4XKslSxkDyl8YeQ284qp3vDx3zP0ZL/view?usp=sharing)
  * Процесс обучения для DRCT + [MambaIR](https://github.com/csguoh/MambaIR) очень медленный. Если вы заинтересованы, вы можете попробовать оптимизировать/настроить его. Это может быть вызвано рекурсивной природой mambaIR в сочетании с повторным использованием карты признаков из DRCT, что приводит к слишком медленной скорости обучения (это также может быть проблема с используемым нами оборудованием или версией пакета).
  * Мы пытаемся объединить DRCT с SS2D в mambaIR. Однако версия CUDA нашего GPU не может быть обновлена до последней версии, что приводит к трудностям при установке пакета и оптимизации скорости обучения. Поэтому мы не планируем продолжать исправлять MambaDRCT. Если вы заинтересованы, можете использовать этот код.
- ✅ 09.06.2024: Выпущена предварительно обученная модель DRCT. [модельный зоопарк](https://drive.google.com/drive/folders/1QJHdSfo-0eFNb96i8qzMJAPw31u9qZ7U?usp=sharing)
- ✅ 11.06.2024: Мы получили большое количество запросов на выпуск предварительно обученных моделей и записей обучения из ImageNet для нескольких последующих приложений, пожалуйста, обратитесь по следующим ссылкам:

[[Журнал обучения на ImageNet]](https://drive.google.com/file/d/1kl2r9TbQ8TR-sOdzvCcOZ9eqNsmIldGH/view?usp=drive_link) [[Предварительно обученные веса (без тонкой настройки на DF2K)]](https://drive.google.com/file/d/1uLGwmSko9uF82X4OPOMw3xfM3stlnYZ-/view?usp=sharing)

- ✅ 12.06.2024: DRCT был выбран для устной презентации в NTIRE!
- ✅ 14.06.2024: Мы получили большое количество запросов на выпуск карт признаков и визуализации LAM, пожалуйста, обратитесь к *./Visualization/*.
- 24.06.2024: DRCT-v2 находится в разработке.
- ✅ 08.07.2024: Обновлен файл вывода (с половинной точностью), огромное спасибо @zelenooki87!
- ✅ 04.12.2024: Обновлена несоответствующая часть после повторного обучения (SRx2). (См. Arxiv)
- ✅ 04.12.2024: Обновлены ошибки описания архитектуры модели. (См. Arxiv)
- ✅ 04.12.2024: Обновлен Real-DRCT-GAN на Google Drive.

---

## Окружение

- [PyTorch >= 2.7.1](https://pytorch.org/)
- [BasicSR == 1.4.2](https://github.com/XPixelGroup/BasicSR/blob/master/INSTALL.md)

### Установка

git clone [https://github.com/igoradmtg/DRCT.git](https://github.com/igoradmtg/DRCT.git)
conda create --name drct python=3.13.5 -y
conda activate drct

# CUDA 11.6

conda install pytorch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 cudatoolkit=11.6 -c pytorch -c conda-forge
cd DRCT
pip install -r requirements.txt
python setup.py develop

## Как выполнить вывод на вашем собственном наборе данных?

python inference.py --input\_dir [input\_dir ] --output\_dir [input\_dir ] --model\_path[model\_path]

## Как тестировать

- Обратитесь к `./options/test` для файла конфигурации тестируемой модели и подготовьте тестовые данные и предварительно обученную модель.
- Затем запустите следующий код (на примере `DRCT_SRx4_ImageNet-pretrain.pth`):

python drct/test.py -opt options/test/DRCT\_SRx4\_ImageNet-pretrain.yml

Результаты тестирования будут сохранены в папке `./results`.

- Обратитесь к `./options/test/DRCT_SRx4_ImageNet-LR.yml` для **вывода** без изображения истинной метки.

**Обратите внимание, что также предусмотрен плиточный режим для ограниченной памяти GPU при тестировании. Вы можете изменить конкретные настройки плиточного режима в своей пользовательской опции тестирования, обратившись к `./options/test/DRCT_tile_example.yml`.**

## Как обучать

- Обратитесь к `./options/train` для файла конфигурации модели для обучения.
- Подготовку обучающих данных можно найти на [этой странице](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md). Набор данных ImageNet можно загрузить с [официального сайта](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php).
- Данные для валидации можно загрузить на [этой странице](https://github.com/ChaofWang/Awesome-Super-Resolution/blob/master/dataset.md).
- Команда для обучения выглядит следующим образом:

CUDA\_VISIBLE\_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc\_per\_node=8 --master\_port=4321 drct/train.py -opt options/train/train\_DRCT\_SRx2\_from\_scratch.yml --launcher pytorch

Журналы обучения и веса будут сохранены в папке `./experiments`.


## Цитирование

Если наша работа полезна для ваших исследований, пожалуйста, сошлитесь на нее. Спасибо!

#### BibTeX

@misc{hsu2024drct,
title={DRCT: Saving Image Super-resolution away from Information Bottleneck},
author = {Hsu, Chih-Chung and Lee, Chia-Ming and Chou, Yi-Shiuan},
year={2024},
eprint={2404.00722},
archivePrefix={arXiv},
primaryClass={cs.CV}
}
@InProceedings{Hsu\_2024\_CVPR,
author    = {Hsu, Chih-Chung and Lee, Chia-Ming and Chou, Yi-Shiuan},
title     = {DRCT: Saving Image Super-Resolution Away from Information Bottleneck},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month     = {June},
year      = {2024},
pages     = {6133-6142}
}

## Благодарности

Часть нашей работы была облегчена фреймворками [HAT](https://github.com/XPixelGroup/HAT), [SwinIR](https://github.com/JingyunLiang/SwinIR) и [LAM](https://github.com/XPixelGroup/X-Low-level-Interpretation), и мы благодарны за их выдающийся вклад.

Часть нашей работы выполнена при участии @zelenooki87, спасибо за ваш большой вклад и предложения!

Особая благодарность [Phhofm](https://github.com/Phhofm) за предоставление модели 4xRealWebPhoto_v4_drct-l, которая значительно улучшила наши возможности обработки изображений. Модель доступна по ссылке [Phhofm/models](https://github.com/Phhofm/models/releases/tag/4xRealWebPhoto_v4_drct-l).


## Контакты

Если у вас есть вопросы, пожалуйста, напишите zuw408421476@gmail.com, чтобы обсудить их с автором.
