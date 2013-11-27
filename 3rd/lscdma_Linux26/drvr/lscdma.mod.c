#include <linux/module.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

MODULE_INFO(vermagic, VERMAGIC_STRING);

struct module __this_module
__attribute__((section(".gnu.linkonce.this_module"))) = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

static const struct modversion_info ____versions[]
__used
__attribute__((section("__versions"))) = {
	{ 0xa8c16cf3, "module_layout" },
	{ 0x54554948, "kobject_put" },
	{ 0x1d26dbf2, "cdev_del" },
	{ 0x352091e6, "kmalloc_caches" },
	{ 0x640a934, "pci_bus_read_config_byte" },
	{ 0xd2b09ce5, "__kmalloc" },
	{ 0x1606bd6d, "cdev_init" },
	{ 0xf5893abf, "up_read" },
	{ 0x4c4fef19, "kernel_stack" },
	{ 0xadaabe1b, "pv_lock_ops" },
	{ 0x15692c87, "param_ops_int" },
	{ 0xc5ac94d0, "dev_set_drvdata" },
	{ 0xc8b57c27, "autoremove_wake_function" },
	{ 0x317830a3, "dma_set_mask" },
	{ 0x1f3ca42, "pci_disable_device" },
	{ 0xf22449ae, "down_interruptible" },
	{ 0x6f2e6f37, "remove_proc_entry" },
	{ 0x152aeb08, "device_destroy" },
	{ 0x8f52a40d, "kobject_set_name" },
	{ 0x3fec048f, "sg_next" },
	{ 0x4632e3d8, "x86_dma_fallback_dev" },
	{ 0xeae3dfd6, "__const_udelay" },
	{ 0x632faaec, "pci_release_regions" },
	{ 0x7485e15e, "unregister_chrdev_region" },
	{ 0x91715312, "sprintf" },
	{ 0x57a6ccd0, "down_read" },
	{ 0xf432dd3d, "__init_waitqueue_head" },
	{ 0x4f8b5ddb, "_copy_to_user" },
	{ 0xe28af57f, "pci_set_master" },
	{ 0xfb578fc5, "memset" },
	{ 0x8f64aa4, "_raw_spin_unlock_irqrestore" },
	{ 0x571ab46f, "current_task" },
	{ 0x27e1a049, "printk" },
	{ 0xfaef0ed, "__tasklet_schedule" },
	{ 0xa1c76e0a, "_cond_resched" },
	{ 0x9166fada, "strncpy" },
	{ 0xb4390f9a, "mcount" },
	{ 0x9545af6d, "tasklet_init" },
	{ 0x6a517590, "device_create" },
	{ 0x2072ee9b, "request_threaded_irq" },
	{ 0xc4556ebd, "cdev_add" },
	{ 0x42c8de35, "ioremap_nocache" },
	{ 0xe660bd22, "pci_bus_read_config_word" },
	{ 0x5acc377, "pci_bus_read_config_dword" },
	{ 0xa31807da, "get_user_pages" },
	{ 0x1000e51, "schedule" },
	{ 0x77ba54e8, "create_proc_entry" },
	{ 0x3ba086f3, "pci_unregister_driver" },
	{ 0x783c7933, "kmem_cache_alloc_trace" },
	{ 0xd52bf1ce, "_raw_spin_lock" },
	{ 0x9327f5ce, "_raw_spin_lock_irqsave" },
	{ 0xcf21d241, "__wake_up" },
	{ 0x37a0cba, "kfree" },
	{ 0x5b6c9815, "remap_pfn_range" },
	{ 0xc8450627, "pci_request_regions" },
	{ 0x5c8b5ce8, "prepare_to_wait" },
	{ 0xb3efa0c4, "pci_disable_msi" },
	{ 0xedc03953, "iounmap" },
	{ 0x71e3cecb, "up" },
	{ 0x7efeadd, "__pci_register_driver" },
	{ 0xec7e91a4, "put_page" },
	{ 0x940df3fb, "class_destroy" },
	{ 0xfa66f77c, "finish_wait" },
	{ 0x1088d836, "pci_enable_msi_block" },
	{ 0xe3e5d565, "pci_enable_device" },
	{ 0x4f6b400b, "_copy_from_user" },
	{ 0x8037428e, "__class_create" },
	{ 0x70c55916, "dev_get_drvdata" },
	{ 0xfd5f002d, "dma_ops" },
	{ 0x29537c9e, "alloc_chrdev_region" },
	{ 0xf20dabd8, "free_irq" },
};

static const char __module_depends[]
__used
__attribute__((section(".modinfo"))) =
"depends=";

MODULE_ALIAS("pci:v00001204d00005303sv00001204sd00003040bc*sc*i*");
MODULE_ALIAS("pci:v00001204d0000E250sv00001204sd00003040bc*sc*i*");
MODULE_ALIAS("pci:v00001204d0000EC30sv00001204sd00003040bc*sc*i*");

MODULE_INFO(srcversion, "884E22A3406176D3EC134A5");
