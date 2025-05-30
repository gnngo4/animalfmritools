# Ubuntu 22.04 LTS - Jammy
ARG BASE_IMAGE=ubuntu:jammy-20240125

FROM ${BASE_IMAGE} AS downloader
RUN echo "2025.05.10"
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    binutils \
                    bzip2 \
                    ca-certificates \
                    curl \
                    wget \
                    unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Python set-up with uv
FROM ghcr.io/astral-sh/uv:0.7.3-python3.13-bookworm-slim AS python_builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /src
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev
ADD . /src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Get Python from a Python image
FROM python:3.13-slim-bookworm AS python_base

# ANTs - v2.6.0
FROM downloader AS ants_builder
RUN mkdir -p /opt/ants \
    && wget -O /tmp/ants.zip "https://github.com/ANTsX/ANTs/releases/download/v2.6.0/ants-2.6.0-ubuntu-22.04-X64-gcc.zip" \
    && unzip -d /opt/ants /tmp/ants.zip \
    && mv /opt/ants/ants-2.6.0/* /opt/ants/ \
    && rm /opt/ants/ants-2.6.0 -rf

# FSL 6.0.5.1
FROM downloader AS fsl_builder
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

# Freesurfer
FROM downloader AS freesurfer_builder
COPY docker/files/freesurfer7.4.1-exclude.txt /usr/local/etc/freesurfer7.4.1-exclude.txt
RUN curl -sSL https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.4.1/freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz \
     | tar zxv --no-same-owner -C /opt --exclude-from=/usr/local/etc/freesurfer7.4.1-exclude.txt

# AFNI
FROM downloader AS afni_builder
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
FROM downloader AS workbench_builder
RUN mkdir /opt/workbench && \
    curl -sSLO https://www.humanconnectome.org/storage/app/media/workbench/workbench-linux64-v1.5.0.zip && \
    unzip workbench-linux64-v1.5.0.zip -d /opt && \
    rm workbench-linux64-v1.5.0.zip && \
    rm -rf /opt/workbench/libs_linux64_software_opengl /opt/workbench/plugins_linux64 && \
    strip --remove-section=.note.ABI-tag /opt/workbench/libs_linux64/libQt5Core.so.5

# Convert3d 1.4.0
FROM downloader AS c3d_builder
RUN mkdir /opt/convert3d && \
    curl -fsSL --retry 5 https://sourceforge.net/projects/c3d/files/c3d/Experimental/c3d-1.4.0-Linux-gcc64.tar.gz/download \
    | tar -xz -C /opt/convert3d --strip-components 1

# Final image without 
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive

# Install Python in the final stage
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        libssl3 \
        libexpat1 \
        ca-certificates \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

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

# Copy Python from the official Python image
COPY --from=python_base /usr/local/bin /usr/local/bin
COPY --from=python_base /usr/local/lib /usr/local/lib
COPY --from=python_base /usr/local/include /usr/local/include

# Copy artifacts from build stages
COPY --from=python_builder /src /src
COPY --from=ants_builder /opt/ants /opt/ants
COPY --from=fsl_builder /opt/fsl /opt/fsl
COPY --from=freesurfer_builder /opt/freesurfer /opt/freesurfer
COPY docker/files/freesurfer_license.txt /opt/freesurfer/license.txt
COPY --from=afni_builder /opt/afni-latest /opt/afni-latest
COPY --from=workbench_builder /opt/workbench /opt/workbench
COPY --from=c3d_builder /opt/convert3d/bin/c3d_affine_tool /usr/bin/c3d_affine_tool

# ANTs config
ENV PATH="/opt/ants/bin:$PATH" \
    ANTSPATH="/opt/ants"

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

# Freesurfer config
ENV OS="Linux" \
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

ENV PATH="/src/.venv/bin:$PATH"

CMD ["python", "/src/animalfmritools/cli/run.py"]