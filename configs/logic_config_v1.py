import easydict as ed

opt_logic = ed.EasyDict()

opt_logic.image_manage = ed.EasyDict()
opt_logic.image_manage.tail = '.png'
opt_logic.image_manage.gt_tail = '_mask_gt.png'
opt_logic.image_manage.pattern = '^[0-9]{1,4}' + opt_logic.image_manage.tail + '$'
opt_logic.image_manage.gt_pattern = '^[0-9]{1,4}' + opt_logic.image_manage.gt_tail + '$'
opt_logic.image_manage.img_name_example = '804.png'
opt_logic.image_manage.gt_img_name_example = '804_mask_gt.png'
opt_logic.image_manage.splitter = '.'
opt_logic.image_manage.gt_splitter = '_mask_gt.'
opt_logic.image_manage.index = -2
opt_logic.image_manage.scale_size_H = 1000
opt_logic.image_manage.scale_size_W = 1000



