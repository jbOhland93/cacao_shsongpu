#ifndef SGR_RECORDER_HPP
#define SGR_RECORDER_HPP

// A class for recording SHS references
class SGR_Recorder
{
public:
    SGR_Recorder(float pxSize, float mlaPitch, float mlaDist);
    const char* getStateDescription();

private:
    float mPxSize;
    float mMlaPitch;
    float mMlaDist;
    enum RECSTATE {INIT,RECORD,FINISH};
    RECSTATE mState = RECSTATE::INIT;
};

#endif // SGR_RECORDER_HPP