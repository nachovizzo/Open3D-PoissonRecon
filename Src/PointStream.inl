/*
Copyright (c) 2006, Michael Kazhdan and Matthew Bolitho
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list
of conditions and the following disclaimer. Redistributions in binary form must
reproduce the above copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided with the
distribution.

Neither the name of the Johns Hopkins University nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

////////////////////////////
// MemoryInputPointStream //
////////////////////////////
template <class Real, unsigned int Dim>
MemoryInputPointStream<Real, Dim>::MemoryInputPointStream(
        size_t pointCount, const Point<Real, Dim>* points) {
    _points = points, _pointCount = pointCount, _current = 0;
}
template <class Real, unsigned int Dim>
MemoryInputPointStream<Real, Dim>::~MemoryInputPointStream(void) {
    ;
}
template <class Real, unsigned int Dim>
void MemoryInputPointStream<Real, Dim>::reset(void) {
    _current = 0;
}
template <class Real, unsigned int Dim>
bool MemoryInputPointStream<Real, Dim>::nextPoint(Point<Real, Dim>& p) {
    if (_current >= _pointCount) return false;
    p = _points[_current];
    _current++;
    return true;
}

///////////////////////////
// ASCIIInputPointStream //
///////////////////////////
template <class Real, unsigned int Dim>
ASCIIInputPointStream<Real, Dim>::ASCIIInputPointStream(const char* fileName) {
    _fp = fopen(fileName, "r");
    if (!_fp) ERROR_OUT("Failed to open file for reading: %s", fileName);
}
template <class Real, unsigned int Dim>
ASCIIInputPointStream<Real, Dim>::~ASCIIInputPointStream(void) {
    fclose(_fp);
    _fp = NULL;
}
template <class Real, unsigned int Dim>
void ASCIIInputPointStream<Real, Dim>::reset(void) {
    fseek(_fp, SEEK_SET, 0);
}
template <class Real, unsigned int Dim>
bool ASCIIInputPointStream<Real, Dim>::nextPoint(Point<Real, Dim>& p) {
    float c;
    for (unsigned int d = 0; d < Dim; d++)
        if (fscanf(_fp, " %f ", &c) != 1)
            return false;
        else
            p[d] = (Real)c;
    return true;
}

////////////////////////////
// ASCIIOutputPointStream //
////////////////////////////
template <class Real, unsigned int Dim>
ASCIIOutputPointStream<Real, Dim>::ASCIIOutputPointStream(
        const char* fileName) {
    _fp = fopen(fileName, "w");
    if (!_fp) ERROR_OUT("Failed to open file for writing: %s", fileName);
}
template <class Real, unsigned int Dim>
ASCIIOutputPointStream<Real, Dim>::~ASCIIOutputPointStream(void) {
    fclose(_fp);
    _fp = NULL;
}
template <class Real, unsigned int Dim>
void ASCIIOutputPointStream<Real, Dim>::nextPoint(const Point<Real, Dim>& p) {
    for (unsigned int d = 0; d < Dim; d++) fprintf(_fp, " %f", (float)p[d]);
    fprintf(_fp, "\n");
}

////////////////////////////
// BinaryInputPointStream //
////////////////////////////
template <class Real, unsigned int Dim>
BinaryInputPointStream<Real, Dim>::BinaryInputPointStream(
        const char* fileName) {
    _fp = fopen(fileName, "rb");
    if (!_fp) ERROR_OUT("Failed to open file for reading: %s", fileName);
}
template <class Real, unsigned int Dim>
bool BinaryInputPointStream<Real, Dim>::nextPoint(Point<Real, Dim>& p) {
    return fread(&p, sizeof(Point<Real, Dim>), 1, _fp) == 1;
}

/////////////////////////////
// BinaryOutputPointStream //
/////////////////////////////
template <class Real, unsigned int Dim>
BinaryOutputPointStream<Real, Dim>::BinaryOutputPointStream(
        const char* fileName) {
    _fp = fopen(fileName, "wb");
    if (!_fp) ERROR_OUT("Failed to open file for writing: %s", fileName);
}
template <class Real, unsigned int Dim>
void BinaryOutputPointStream<Real, Dim>::nextPoint(const Point<Real, Dim>& p) {
    fwrite(&p, sizeof(Point<Real, Dim>), 1, _fp) == 1;
}

////////////////////////////////////
// MemoryInputPointStreamWithData //
////////////////////////////////////
template <class Real, unsigned int Dim, class Data>
MemoryInputPointStreamWithData<Real, Dim, Data>::MemoryInputPointStreamWithData(
        size_t pointCount, const std::pair<Point<Real, Dim>, Data>* points) {
    _points = points, _pointCount = pointCount, _current = 0;
}
template <class Real, unsigned int Dim, class Data>
MemoryInputPointStreamWithData<Real, Dim, Data>::
        ~MemoryInputPointStreamWithData(void) {
    ;
}
template <class Real, unsigned int Dim, class Data>
void MemoryInputPointStreamWithData<Real, Dim, Data>::reset(void) {
    _current = 0;
}
template <class Real, unsigned int Dim, class Data>
bool MemoryInputPointStreamWithData<Real, Dim, Data>::nextPoint(
        Point<Real, Dim>& p, Data& d) {
    if (_current >= _pointCount) return false;
    p = _points[_current].first;
    d = _points[_current].second;
    _current++;
    return true;
}

///////////////////////////////////
// ASCIIInputPointStreamWithData //
///////////////////////////////////
template <class Real, unsigned int Dim, class Data>
ASCIIInputPointStreamWithData<Real, Dim, Data>::ASCIIInputPointStreamWithData(
        const char* fileName, void (*ReadData)(FILE*, Data&))
    : _ReadData(ReadData) {
    _fp = fopen(fileName, "r");
    if (!_fp) ERROR_OUT("Failed to open file for reading: ", fileName);
}
template <class Real, unsigned int Dim, class Data>
ASCIIInputPointStreamWithData<Real, Dim, Data>::~ASCIIInputPointStreamWithData(
        void) {
    fclose(_fp);
    _fp = NULL;
}
template <class Real, unsigned int Dim, class Data>
void ASCIIInputPointStreamWithData<Real, Dim, Data>::reset(void) {
    fseek(_fp, SEEK_SET, 0);
}
template <class Real, unsigned int Dim, class Data>
bool ASCIIInputPointStreamWithData<Real, Dim, Data>::nextPoint(
        Point<Real, Dim>& p, Data& d) {
    float c;
    for (unsigned int dd = 0; dd < Dim; dd++)
        if (fscanf(_fp, " %f ", &c) != 1)
            return false;
        else
            p[dd] = c;
    _ReadData(_fp, d);
    return true;
}

////////////////////////////////////
// ASCIIOutputPointStreamWithData //
////////////////////////////////////
template <class Real, unsigned int Dim, class Data>
ASCIIOutputPointStreamWithData<Real, Dim, Data>::ASCIIOutputPointStreamWithData(
        const char* fileName, void (*WriteData)(FILE*, const Data&))
    : _WriteData(WriteData) {
    _fp = fopen(fileName, "w");
    if (!_fp) ERROR_OUT("Failed to open file for writing: %s", fileName);
}
template <class Real, unsigned int Dim, class Data>
ASCIIOutputPointStreamWithData<Real, Dim, Data>::
        ~ASCIIOutputPointStreamWithData(void) {
    fclose(_fp);
    _fp = NULL;
}
template <class Real, unsigned int Dim, class Data>
void ASCIIOutputPointStreamWithData<Real, Dim, Data>::nextPoint(
        const Point<Real, Dim>& p, const Data& d) {
    for (unsigned int d = 0; d < Dim; d++) fprintf(_fp, " %f", (float)p[d]);
    fprintf(_fp, " ");
    _WriteData(_fp, d);
    fprintf(_fp, "\n");
}

////////////////////////////////////
// BinaryInputPointStreamWithData //
////////////////////////////////////
template <class Real, unsigned int Dim, class Data>
BinaryInputPointStreamWithData<Real, Dim, Data>::BinaryInputPointStreamWithData(
        const char* fileName, void (*ReadData)(FILE*, Data&))
    : _ReadData(ReadData) {
    _fp = fopen(fileName, "rb");
    if (!_fp) ERROR_OUT("Failed to open file for reading: ", fileName);
}
template <class Real, unsigned int Dim, class Data>
bool BinaryInputPointStreamWithData<Real, Dim, Data>::nextPoint(
        Point<Real, Dim>& p, Data& d) {
    if (fread(&p, sizeof(Point<Real, Dim>), 1, _fp) == 1) {
        _ReadData(_fp, d);
        return true;
    } else
        return false;
}

/////////////////////////////////////
// BinaryOutputPointStreamWithData //
/////////////////////////////////////
template <class Real, unsigned int Dim, class Data>
BinaryOutputPointStreamWithData<Real, Dim, Data>::
        BinaryOutputPointStreamWithData(const char* fileName,
                                        void (*WriteData)(FILE*, const Data&))
    : _WriteData(WriteData) {
    _fp = fopen(fileName, "wb");
    if (!_fp) ERROR_OUT("Failed to open file for writing: ", fileName);
}
template <class Real, unsigned int Dim, class Data>
void BinaryOutputPointStreamWithData<Real, Dim, Data>::nextPoint(
        const Point<Real, Dim>& p, const Data& d) {
    fwrite(&p, sizeof(Point<Real, Dim>), 1, _fp);
    _WriteData(_fp, d);
}
