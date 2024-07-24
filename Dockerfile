ARG BASE_IMAGE=ubuntu:jammy-20240627.1

### BUILD python packages
FROM python:slim AS src
WORKDIR /src
# for compiling traits<6.4 wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN pip install --no-cache-dir pipenv
COPY Pipfile Pipfile.lock /src/
RUN pipenv install --system --deploy --ignore-pipfile

# From fmriprep - base downloader
FROM ${BASE_IMAGE} as downloader
RUN echo "2024.07.22"
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    binutils \
                    bzip2 \
                    ca-certificates \
                    curl \
                    unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# FSL 6.0.5.1
FROM downloader as fsl
RUN mkdir -p /opt/fsl \
    && curl -fsSL --retry 5 https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-6.0.5.1-centos7_64.tar.gz \
    | tar -xz -C /opt/fsl --strip-components 1 \
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
    && find /opt/fsl/bin -type f -not \( \
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
    && find /opt/fsl/data/standard -type f -not -name "MNI152_T1_2mm_brain.nii.gz" -delete

# ANTs
FROM downloader as ants
RUN mkdir -p /opt/ants \
    && curl -sSL "https://dl.dropbox.com/s/gwf51ykkk5bifyj/ants-Linux-centos6_x86_64-v2.3.4.tar.gz" \
    | tar -xzC /opt/ants --strip-components 1

# Freesurfer
FROM downloader as freesurfer
COPY docker/files/freesurfer7.4.1-exclude.txt /usr/local/etc/freesurfer7.4.1-exclude.txt
RUN curl -sSL https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.4.1/freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz \
     | tar zxv --no-same-owner -C /opt --exclude-from=/usr/local/etc/freesurfer7.4.1-exclude.txt

# AFNI
FROM downloader as afni
# Bump the date to current to update AFNI
RUN echo "2023.07.20"
RUN mkdir -p /opt/afni-latest \
    && curl -fsSL --retry 5 https://afni.nimh.nih.gov/pub/dist/tgz/linux_openmp_64.tgz \
    | tar -xz -C /opt/afni-latest --strip-components 1 \
    --exclude "linux_openmp_64/*.gz" \
    --exclude "linux_openmp_64/funstuff" \
    --exclude "linux_openmp_64/shiny" \
    --exclude "linux_openmp_64/afnipy" \
    --exclude "linux_openmp_64/lib/RetroTS" \
    --exclude "linux_openmp_64/lib_RetroTS" \
    --exclude "linux_openmp_64/meica.libs" \
    # Keep only what we use
    && find /opt/afni-latest -type f -not \( \
        -name "3dTshift" -or \
        -name "3dUnifize" -or \
        -name "3dAutomask" -or \
        -name "3dvolreg" \) -delete

# Connectome Workbench 1.5.0
FROM downloader as workbench
RUN mkdir /opt/workbench && \
    curl -sSLO https://www.humanconnectome.org/storage/app/media/workbench/workbench-linux64-v1.5.0.zip && \
    unzip workbench-linux64-v1.5.0.zip -d /opt && \
    rm workbench-linux64-v1.5.0.zip && \
    rm -rf /opt/workbench/libs_linux64_software_opengl /opt/workbench/plugins_linux64 && \
    strip --remove-section=.note.ABI-tag /opt/workbench/libs_linux64/libQt5Core.so.5

# Convert3d 1.4.0
FROM downloader as c3d
RUN mkdir /opt/convert3d && \
    curl -fsSL --retry 5 https://sourceforge.net/projects/c3d/files/c3d/Experimental/c3d-1.4.0-Linux-gcc64.tar.gz/download \
    | tar -xz -C /opt/convert3d --strip-components 1

### MAIN STAGE
FROM ${BASE_IMAGE} AS animalfmritools
# Configure apt
ENV DEBIAN_FRONTEND="noninteractive" \
    LANG="C.UTF-8" \
    LC_ALL="C.UTF-8"
# Some baseline tools; bc is needed for FreeSurfer, so don't drop it
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    bc \
                    ca-certificates \
                    curl \
                    git \
                    gnupg \
                    lsb-release \
                    netbase \
                    xvfb && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# Configure PPAs for libpng12 and libxp6
RUN GNUPGHOME=/tmp gpg --keyserver hkps://keyserver.ubuntu.com --no-default-keyring --keyring /usr/share/keyrings/linuxuprising.gpg --recv 0xEA8CACC073C3DB2A \
    && GNUPGHOME=/tmp gpg --keyserver hkps://keyserver.ubuntu.com --no-default-keyring --keyring /usr/share/keyrings/zeehio.gpg --recv 0xA1301338A3A48C4A \
    && echo "deb [signed-by=/usr/share/keyrings/linuxuprising.gpg] https://ppa.launchpadcontent.net/linuxuprising/libpng12/ubuntu jammy main" > /etc/apt/sources.list.d/linuxuprising.list \
    && echo "deb [signed-by=/usr/share/keyrings/zeehio.gpg] https://ppa.launchpadcontent.net/zeehio/libxp/ubuntu jammy main" > /etc/apt/sources.list.d/zeehio.list

# Dependencies for FSL
RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
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
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Dependencies for AFNI; requires a discontinued multiarch-support package from bionic (18.04)
RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           ed \
           gsl-bin \
           libglib2.0-0 \
           libglu1-mesa-dev \
           libglw1-mesa \
           libgomp1 \
           libjpeg62 \
           libpng12-0 \
           libxm4 \
           libxp6 \
           netpbm \
           tcsh \
           xfonts-base \
           xvfb \
    && curl -sSL --retry 5 -o /tmp/multiarch.deb http://archive.ubuntu.com/ubuntu/pool/main/g/glibc/multiarch-support_2.27-3ubuntu1.5_amd64.deb \
    && dpkg -i /tmp/multiarch.deb \
    && rm /tmp/multiarch.deb \
    && apt-get install -f \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && gsl2_path="$(find / -name 'libgsl.so.19' || printf '')" \
    && if [ -n "$gsl2_path" ]; then \
         ln -sfv "$gsl2_path" "$(dirname $gsl2_path)/libgsl.so.0"; \
    fi \
    && ldconfig

# Install files from stages
COPY --from=fsl /opt/fsl /opt/fsl
COPY --from=ants /opt/ants /opt/ants
COPY --from=freesurfer /opt/freesurfer /opt/freesurfer
COPY docker/files/freesurfer_license.txt /opt/freesurfer/license.txt
COPY --from=afni /opt/afni-latest /opt/afni-latest
COPY --from=workbench /opt/workbench /opt/workbench
COPY --from=c3d /opt/convert3d/bin/c3d_affine_tool /usr/bin/c3d_affine_tool

# FSL config
ENV FSLDIR="/opt/fsl" \
    PATH="/opt/fsl/bin:$PATH" \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    FSLLOCKDIR="" \
    FSLMACHINELIST="" \
    FSLREMOTECALL="" \
    FSLGECUDAQ="cuda.q" \
    LD_LIBRARY_PATH="/opt/fsl/lib:$LD_LIBRARY_PATH"
COPY docker/bin/imcp /opt/fsl/bin/imcp

# ANTs config
ENV PATH="/opt/ants:$PATH" \
    ANTSPATH="/opt/ants"

# Simulate SetUpFreeSurfer.sh
ENV FSL_DIR="/opt/fsl" \
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

# AFNI config
ENV PATH="/opt/afni-latest:$PATH" \
    AFNI_IMSAVE_WARNINGS="NO" \
    AFNI_PLUGINPATH="/opt/afni-latest"

# Workbench config
ENV PATH="/opt/workbench/bin_linux64:$PATH" \
    LD_LIBRARY_PATH="/opt/workbench/lib_linux64:$LD_LIBRARY_PATH"

RUN ldconfig

# Set working directory
WORKDIR /opt
# Copy the /usr/local directory from the src stage
# This ensures Python ensures that the Python environment
# in the final stage is identical to the build stage
COPY --from=src /usr/local /usr/local
# Add animalfmritools
ADD animalfmritools /opt/animalfmritools/animalfmritools
ENTRYPOINT ["python3", "/opt/animalfmritools/animalfmritools/cli/run.py"]
