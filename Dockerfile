FROM ubuntu:jammy-20230308

ENV DEBIAN_FRONTEND="noninteractive"

#RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update -y && apt upgrade -y && \
RUN apt-get update -y && apt upgrade -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        curl \
        unzip \
        tree \
        libxt-dev && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*
    
# Installing freesurfer
COPY docker/files/freesurfer7.2-exclude.txt /usr/local/etc/freesurfer7.2-exclude.txt
RUN curl -sSL https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.2.0/freesurfer-linux-ubuntu18_amd64-7.2.0.tar.gz \
￼    | tar zxv --no-same-owner -C /opt --exclude-from=/usr/local/etc/freesurfer7.2-exclude.txt

# Simulate SetUpFreeSurfer.sh
ENV FSL_DIR="/opt/fsl-6.0.5.1" \
    OS="Linux" \
    FS_OVERRIDE=0 \
    FIX_VERTEX_AREA="" \
    FSF_OUTPUT_FORMAT="nii.gz" \
    FREESURFER_HOME="/opt/freesurfer"
ENV SUBJECTS_DIR="$FREESURFER_HOME/subjects" \
    FUNCTIONALS_DIR="$FREESURFER_HOME/sessions" \
    MNI_DIR="$FREESURFER_HOME/mni" \
    LOCAL_DIR="$FREESURFER_HOME/local" \
    MINC_BIN_DIR="$FREESURFER_HOME/mni/bin" \
    MINC_LIB_DIR="$FREESURFER_HOME/mni/lib" \
    MNI_DATAPATH="$FREESURFER_HOME/mni/data"
ENV PERL5LIB="$MINC_LIB_DIR/perl5/5.8.5" \
    MNI_PERL5LIB="$MINC_LIB_DIR/perl5/5.8.5" \
    PATH="$FREESURFER_HOME/bin:$FREESURFER_HOME/tktools:$MINC_BIN_DIR:$PATH"

# Copy license.txt
COPY docker/files/freesurfer_license.txt /opt/freesurfer/license.txt

# mri_synthstrip 1.3
RUN apt-get update -y && apt upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential python3-dev
RUN fspython -m pip install torch==1.10.2
RUN fspython -m pip install surfa==0.3.3
RUN mkdir -p /opt/synthstrip
COPY --from=freesurfer/synthstrip:1.3 /freesurfer/mri_synthstrip /opt/synthstrip
COPY --from=freesurfer/synthstrip:1.3 /freesurfer/models/synthstrip.*.pt /opt/freesurfer/models/

# Replace recon-all (https://github.com/freesurfer/freesurfer/issues/892)
COPY docker/files/recon-all.edit /opt/freesurfer/bin/recon-all

# FSL 6.0.5.1
RUN apt-get update -y && apt upgrade -y && \
    apt-get install -y --no-install-recommends \
        bc \
        dc \
        file \
        libfontconfig1 \
        libfreetype6 \
        libgl1-mesa-dev \
        libgl1-mesa-dri \
        libglu1-mesa-dev \
        libgomp1 \
        libice6 \
        libxcursor1 \
        libxft2 \
        libxinerama1 \
        libxrandr2 \
        libxrender1 \
        libxt6 \
        libquadmath0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && echo "Downloading FSL ..." \
    && mkdir -p /opt/fsl-6.0.5.1 \
    && curl -fsSL --retry 5 https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-6.0.5.1-centos7_64.tar.gz \
    | tar -xz -C /opt/fsl-6.0.5.1 --strip-components 1 \
    --exclude "fsl/config" \
    --exclude "fsl/data/atlases" \
    --exclude "fsl/data/first" \
    --exclude "fsl/data/mist" \
    --exclude "fsl/data/possum" \
    --exclude "fsl/data/standard/bianca" \
    --exclude "fsl/data/standard/tissuepriors" \
    --exclude "fsl/doc" \
    --exclude "fsl/etc/default_flobs.flobs" \
    --exclude "fsl/etc/fslconf" \
    --exclude "fsl/etc/js" \
    --exclude "fsl/etc/luts" \
    --exclude "fsl/etc/matlab" \
    --exclude "fsl/extras" \
    --exclude "fsl/include" \
    --exclude "fsl/refdoc" \
    --exclude "fsl/src" \
    --exclude "fsl/tcl" \
    --exclude "fsl/bin/FSLeyes" \
    && find /opt/fsl-6.0.5.1/bin -type f -not \( \
        -name "applywarp" -or \
        -name "avscale" -or \
        -name "bet" -or \
        -name "bet2" -or \
        -name "convert_xfm" -or \
        -name "convertwarp" -or \
        -name "fast" -or \
        -name "flirt" -or \
        -name "fsl_regfilt" -or \
        -name "fsl_prepare_fieldmap" -or \
        -name "fslhd" -or \
        -name "fslinfo" -or \
        -name "fslmaths" -or \
        -name "fslmerge" -or \
        -name "fslreorient2std" -or \
        -name "fslroi" -or \
        -name "fslsplit" -or \
        -name "fslstats" -or \
        -name "fslswapdim" -or \
        -name "fslswapdim_exe" -or \
        -name "fslval" -or \
        -name "fugue" -or \
        -name "imtest" -or \
        -name "mcflirt" -or \
        -name "melodic" -or \
        -name "prelude" -or \
        -name "remove_ext" -or \
        -name "susan" -or \
        -name "applytopup" -or \
        -name "tmpnam" -or \
        -name "topup" -or \
        -name "zeropad" \) -delete \
    && find /opt/fsl-6.0.5.1/data/standard -type f -not -name "MNI152_T1_2mm_brain.nii.gz" -delete
ENV FSLDIR="/opt/fsl-6.0.5.1" \
    PATH="/opt/fsl-6.0.5.1/bin:$PATH" \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    FSLLOCKDIR="" \
    FSLMACHINELIST="" \
    FSLREMOTECALL="" \
    FSLGECUDAQ="cuda.q" \
    LD_LIBRARY_PATH="/opt/fsl-6.0.5.1/lib:$LD_LIBRARY_PATH"
COPY docker/bin/imcp /opt/fsl-6.0.5.1/bin/imcp

# Convert3D (neurodocker build)
RUN echo "Downloading Convert3D ..." \
    && mkdir -p /opt/convert3d-1.0.0 \
    && curl -fsSL --retry 5 https://sourceforge.net/projects/c3d/files/c3d/1.0.0/c3d-1.0.0-Linux-x86_64.tar.gz/download \
    | tar -xz -C /opt/convert3d-1.0.0 --strip-components 1 \
    --exclude "c3d-1.0.0-Linux-x86_64/lib" \
    --exclude "c3d-1.0.0-Linux-x86_64/share" \
    --exclude "c3d-1.0.0-Linux-x86_64/bin/c3d_gui"
COPY docker/bin/c3d_affine_tool /opt/convert3d-1.0.0/bin/c3d_affine_tool2
ENV C3DPATH="/opt/convert3d-1.0.0" \
    PATH="/opt/convert3d-1.0.0/bin:$PATH"

# AFNI latest (neurodocker build)
ENV PATH="/opt/afni-latest:$PATH" \
    AFNI_PLUGINPATH="/opt/afni-latest"
RUN apt-get update -y && apt upgrade -y && \
    apt-get install -y --no-install-recommends \
        tcsh xfonts-base libssl-dev \
        gsl-bin netpbm gnome-tweaks \
        libjpeg62 xvfb xterm \
        gedit evince eog \
        libglu1-mesa-dev libglw1-mesa \
        libxm4 build-essential \
        libcurl4-openssl-dev libxml2-dev \
        libgfortran-11-dev libgomp1 \
        gnome-terminal nautilus \
        firefox xfonts-100dpi \
        r-base-dev cmake \
        libgdal-dev libopenblas-dev \
        libnode-dev libudunits2-dev && \
    ln -s /usr/lib/x86_64-linux-gnu/libgsl.so.27 /usr/lib/x86_64-linux-gnu/libgsl.so.19 && \
    curl -O https://afni.nimh.nih.gov/pub/dist/bin/misc/@update.afni.binaries && \
    tcsh @update.afni.binaries -package linux_ubuntu_16_64 -do_extras -bindir /opt/afni-latest

# Installing ANTs 2.3.3 (NeuroDocker build)
# Note: the URL says 2.3.4 but it is actually 2.3.3
ENV ANTSPATH="/opt/ants" \
    PATH="/opt/ants:$PATH"
WORKDIR $ANTSPATH
RUN curl -sSL "https://dl.dropbox.com/s/gwf51ykkk5bifyj/ants-Linux-centos6_x86_64-v2.3.4.tar.gz" \
    | tar -xzC $ANTSPATH --strip-components 1

# Workbench
WORKDIR /opt
RUN curl -sSLO https://www.humanconnectome.org/storage/app/media/workbench/workbench-linux64-v1.5.0.zip && \
    unzip workbench-linux64-v1.5.0.zip && \
    rm workbench-linux64-v1.5.0.zip && \
    rm -rf /opt/workbench/libs_linux64_software_opengl /opt/workbench/plugins_linux64 && \
    strip --remove-section=.note.ABI-tag /opt/workbench/libs_linux64/libQt5Core.so.5
    # ABI tags can interfere when running on Singularity

ENV PATH="/opt/workbench/bin_linux64:$PATH" \
    LD_LIBRARY_PATH="/opt/workbench/lib_linux64:$LD_LIBRARY_PATH"

RUN ldconfig

# Python 3.10 and pipenv
RUN apt-get update -y && apt upgrade -y && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip && \
    pip install pipenv

WORKDIR /opt/animalfmritools
COPY ["Pipfile.lock", "/opt/animalfmritools"]
COPY ["Pipfile", "/opt/animalfmritools"]
ADD ["animalfmritools", "/opt/animalfmritools/animalfmritools"]
RUN ["pipenv", "install", "--deploy", "--system", "--ignore-pipfile"]

WORKDIR /opt

ENTRYPOINT ["python3", "/opt/animalfmritools/animalfmritools/cli/run.py"]
