import easydict as ed

opt_logic = ed.EasyDict()

opt_logic.image_manage = ed.EasyDict()
opt_logic.image_manage.tail = '.png'
opt_logic.image_manage.pattern = '^[0-9]{1,4}' + opt_logic.image_manage.tail + '$'
opt_logic.image_manage.img_name_example = '804.png'


