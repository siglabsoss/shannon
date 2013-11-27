#!/bin/sh

#----------------------------------------------------------------------
# This shell script installs the the Lattice lscpcie2 Driver Module
# into the Linux kernel.
#
# The major number is dynamically assigned.
# some how we are going to get an entry into the /dev via udev

# dynamically assigned Major number for lscpcie2.ko
# Minor nmbers identify the board type, instance and BAR.
# This version creates 4 SC entries and 4 EC entries, each with 6
# BARs. The base module name is "lscpcie2".  The extension is ".ko"
# which represents a kernel object.
#----------------------------------------------------------------------

module="lscdma"

if [ `whoami` != root ]; then
	echo "ERROR! Must be root to install driver."
	exit 1
fi

echo "Installing driver module "${module}" into kernel."

# Install the driver module (pass any command line args - none expected)
/sbin/insmod -f ./${module}.ko $* || exit 1


mkdir -p /dev/lscdma

# ECP3_DMA_1  ECP3_DMA_1_CB  ECP3_DMA_1_IM

mknod /dev/lscdma/ECP3_DMA_1 c 250 0

mknod /dev/lscdma/ECP3_DMA_1_CB c 250 1
mknod /dev/lscdma/ECP3_DMAB_1 c 250 1

mknod /dev/lscdma/ECP3_DMA_1_IM c 250 2
mknod /dev/lscdma/ECP3_DMAC_1 c 250 2


chmod 666 /dev/lscdma/*





echo "Done."

